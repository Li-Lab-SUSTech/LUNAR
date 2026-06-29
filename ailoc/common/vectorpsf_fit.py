import napari
import numpy as np
import torch
from torch import linalg as LA
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import tifffile
import skimage
import os
import scipy.io as sio

import ailoc.common
import ailoc.simulation


def segment_local_max_beads(params_dict: dict) -> dict:
    # set parameters
    raw_images = ailoc.common.gpu(params_dict['raw_images'].astype(np.float32))
    roi_size = params_dict['roi_size']
    filter_sigma = params_dict['filter_sigma']
    # threshold_rel = params_dict['threshold_rel']
    threshold_abs = params_dict['threshold_abs']

    # transform the raw images to photons
    camera_model = ailoc.simulation.instantiate_camera(params_dict['camera_params_dict'])
    raw_images_photons = ailoc.common.cpu(camera_model.backward(raw_images))

    # remove the nonuniform background for better local max extraction, maybe unnecessary
    bg = skimage.restoration.rolling_ball(np.mean(raw_images_photons, axis=0), radius=roi_size//2)
    images_bg = np.clip(raw_images_photons-bg, a_min=0, a_max=None)

    images_bg_filtered = skimage.filters.gaussian(np.mean(images_bg, axis=0), sigma=filter_sigma)
    peak_coords = skimage.feature.peak_local_max(images_bg_filtered,
                                                 min_distance=int(roi_size*0.5),
                                                 # threshold_rel=threshold_rel,
                                                 threshold_abs=threshold_abs,
                                                 exclude_border=True)

    # Display the results
    fig, ax = plt.subplots()
    img_tmp = ax.imshow(raw_images_photons.mean(axis=0), cmap='gray')
    # img_tmp = ax.imshow(images_bg_filtered, cmap='gray')
    for i in range(peak_coords.shape[0]):
        rect = plt.Rectangle((peak_coords[i, 1] - roi_size//2, peak_coords[i, 0] - roi_size//2), roi_size, roi_size,
                             edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # ax.plot(peak_coords[:, 1], peak_coords[:, 0], 'rx')
    # ax.axis('off')
    plt.colorbar(mappable=img_tmp, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    # segment the beads using the peak coordinates and roi_size
    beads_seg = []
    for peak_coord in peak_coords:
        x, y = peak_coord
        bead_seg = raw_images_photons[:, int(x-roi_size/2):int(x+roi_size/2), int(y-roi_size/2):int(y+roi_size/2)]
        beads_seg.append(bead_seg)

    params_dict['beads_seg'] = beads_seg
    del params_dict['raw_images']

    return params_dict


def zernike_calibrate_3d_beads_stack(params_dict: dict) -> dict:
    # torch.autograd.set_detect_anomaly(True)

    # set parameters
    beads_seg = params_dict['beads_seg']
    roi_size = params_dict['roi_size']
    z_step = params_dict['z_step']
    fit_brightest = params_dict['fit_brightest']
    psf_params_dict = params_dict['psf_params_dict']

    # calibrate the segmented beads, first prepare the parameters
    data_stacked = torch.ceil(torch.clamp(ailoc.common.gpu(np.vstack(beads_seg)), min=1e-6))
    n_beads = len(beads_seg)
    n_zstack = beads_seg[0].shape[0]
    bg_estimated = data_stacked.view(n_zstack * n_beads, -1).min(dim=1).values
    photons_estimated = torch.sum((data_stacked - bg_estimated[:, None, None]), dim=(1, 2)) / 1000
    # fit all beads or only fit the brightest one
    if fit_brightest:
        tmp_brightness = -1
        final_slice = None
        for i in range(n_beads):
            slice_tmp = slice(i * n_zstack, (i + 1) * n_zstack)
            if photons_estimated[slice_tmp].mean() > tmp_brightness:
                tmp_brightness = photons_estimated[slice_tmp].mean()
                final_slice = slice_tmp
        n_beads = 1
        data_stacked = data_stacked[final_slice]

    # first we move the objective away from the beads and step closer
    objstage_prior = ailoc.common.gpu(torch.linspace(n_zstack//2*z_step, -(n_zstack//2*z_step), n_zstack))
    psf_torch_fitted = ailoc.simulation.VectorPSFTorch(psf_params_dict, req_grad=True, data_type=torch.float32)

    # n_beads parameters
    x_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    y_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    objstage_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))

    # n_zstack * n_beads parameters
    photons_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))
    bg_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))

    # # use shared photons and bg for all z positions
    # photons_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(1 * n_beads)))
    # bg_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(1 * n_beads)))

    # initialize the parameters
    x_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size-1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       (roi_size-1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       roi_size))
    y_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size-1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       (roi_size-1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       roi_size))
    x_mesh, y_mesh = torch.meshgrid(x_image_linspace, y_image_linspace, indexing='xy')
    with torch.no_grad():
        bg_fitted += data_stacked.view(n_zstack * n_beads, -1).min(dim=1).values
        photons_fitted += torch.sum((data_stacked-bg_fitted[:, None, None]), dim=(1, 2))/1000

        # # use shared photons and bg for all z positions
        # bg_fitted += data_stacked.view(n_zstack * n_beads, -1).min()
        # photons_fitted += torch.mean(torch.sum((data_stacked - bg_fitted[:, None, None]), dim=(1, 2))) / 1000

        for i in range(n_beads):
            slice_tmp = slice(i*n_zstack, (i+1)*n_zstack)
            x_fitted[i] += torch.sum(x_mesh*torch.mean(data_stacked[slice_tmp], dim=0))/(photons_fitted.mean()*1000)
            y_fitted[i] += torch.sum(y_mesh*torch.mean(data_stacked[slice_tmp], dim=0))/(photons_fitted.mean()*1000)

    # # AdamW
    # tolerance = 1e-5
    # old_loss = 1e10
    # optimizer = torch.optim.AdamW([
    #     psf_torch_fitted.zernike_coef,
    #     x_fitted,
    #     y_fitted,
    #     objstage_fitted,
    #     photons_fitted,
    #     bg_fitted
    #     ], lr=5)
    #
    # for iterations in range(10000):
    #     x_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     y_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     z_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     objstage_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     for i in range(n_beads):
    #         x_tmp[i * n_zstack:(i + 1) * n_zstack] = x_fitted[i]
    #         y_tmp[i * n_zstack:(i + 1) * n_zstack] = y_fitted[i]
    #         objstage_tmp[i * n_zstack:(i + 1) * n_zstack] = objstage_fitted[i] + objstage_prior
    #
    #     photons_tmp = torch.clamp(photons_fitted*1000, min=0)
    #     bg_tmp = torch.clamp(bg_fitted, min=0)
    #
    #     psf_torch_fitted._pre_compute()
    #     model_fitted = psf_torch_fitted.simulate_parallel(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]
    #     model = torch.distributions.Poisson(model_fitted)
    #     loss = -model.log_prob(data_stacked).sum()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f'iter:{iterations}, nll:{loss.item()}')
    #     if abs(old_loss - loss.item()) < tolerance:
    #         break
    #     old_loss = loss.item()

    # LBFGS
    tolerance = 1e-5
    old_loss = 1e10
    optimizer = torch.optim.LBFGS([
         psf_torch_fitted.zernike_coef,
         x_fitted,
         y_fitted,
         objstage_fitted,
         photons_fitted,
         bg_fitted
         ],
         lr=0.5,
         line_search_fn='strong_wolfe')
    # lambda_reg = 0
    lambda_reg = 1e4

    def closure():
        x_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        y_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        z_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        objstage_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        for i in range(n_beads):
            x_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(x_fitted[i],
                                                           min=x_image_linspace.min(),
                                                           max=x_image_linspace.max())
            y_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(y_fitted[i],
                                                           min=y_image_linspace.min(),
                                                           max=y_image_linspace.max())
            objstage_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(objstage_fitted[i],
                                                                  min=objstage_prior.min(),
                                                                  max=objstage_prior.max()) + objstage_prior
        photons_tmp = torch.clamp(photons_fitted*1000, min=0)
        bg_tmp = torch.clamp(bg_fitted, min=0)

        optimizer.zero_grad()

        mu = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]

        assert torch.min(mu) >= 0, 'fitting failed, mu should be positive'
        if photons_fitted.min() <= 0 or bg_fitted.min() <= 0:
            # print('photons or bg <= 0, using torch.clamp and regularization')
            indices_photons = torch.nonzero(photons_fitted <= 0)
            indices_bg = torch.nonzero(bg_fitted <= 0)
            model = torch.distributions.Poisson(mu)

            # # torch api for poisson loss and regularization v2
            # loss = -model.log_prob(data_stacked).sum() + \
            #        1 * ((photons_fitted[indices_photons]**2).sum() + (bg_fitted[indices_bg]**2).sum())

            # torch api for poisson loss and regularization
            loss = -model.log_prob(data_stacked).sum() - \
                   lambda_reg * (photons_fitted[indices_photons].sum() + bg_fitted[indices_bg].sum())

            # # analytical poisson loss form and regularization
            # loss = -torch.sum(data_stacked * torch.log(mu) - mu - torch.lgamma(data_stacked + 1)) - \
            #        lambda_reg*(photons_fitted[indices_photons].sum() + bg_fitted[indices_bg].sum())
        else:
            model = torch.distributions.Poisson(mu)
            loss = -model.log_prob(data_stacked).sum()
            # loss = -torch.sum(data_stacked * torch.log(mu) - mu - torch.lgamma(data_stacked + 1))

        loss.backward()
        return loss.detach()

    for iteration in range(75):
        loss = closure()
        print(f'iter:{iteration}, nll:{loss.item()}')
        if abs(old_loss - loss.item()) < tolerance:
            break
        old_loss = loss.item()
        optimizer.step(closure)

    # prepare the results
    x_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    y_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    z_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    objstage_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    for i in range(n_beads):
        x_tmp[i * n_zstack:(i + 1) * n_zstack] = x_fitted[i]
        y_tmp[i * n_zstack:(i + 1) * n_zstack] = y_fitted[i]
        objstage_tmp[i * n_zstack:(i + 1) * n_zstack] = objstage_fitted[i] + objstage_prior
    photons_tmp = photons_fitted * 1000
    bg_tmp = bg_fitted

    mu = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]

    psf_params_fitted = psf_params_dict.copy()
    psf_params_fitted['zernike_coef'] = psf_torch_fitted.zernike_coef.detach().cpu().numpy()

    calib_params_dict = params_dict.copy()

    data_measured = ailoc.common.cpu(data_stacked)
    data_fitted = ailoc.common.cpu(mu)

    results = {'calib_params_dict': calib_params_dict, 'psf_params_fitted': psf_params_fitted,
               'data_measured': data_measured, 'data_fitted': data_fitted}

    # plot the results
    font_size = 10

    rrse = np.linalg.norm((data_fitted - data_measured).flatten(), ord=2) / \
           np.linalg.norm(data_measured.flatten(), ord=2) * 100

    nmol = n_zstack
    if nmol <= 35:
        ncolumns = 5
        nrows = int(np.floor(nmol / ncolumns))
        nidx = np.arange(nmol)
    else:
        ncolumns = 5
        nrows = 7
        nidx = np.linspace(0, nmol-1, ncolumns*nrows, dtype=int)

    # plot the data, model, error
    figure, ax = plt.subplots(nrows, ncolumns, constrained_layout=True, figsize=(10, 8))
    i_slice = 0
    for i in range(nrows):
        for j in range(ncolumns):
            beads_tmp = data_measured[nidx[i_slice]]
            model_tmp = data_fitted[nidx[i_slice]]
            mismatch_tmp = beads_tmp - model_tmp
            image_tmp = np.concatenate([beads_tmp, model_tmp, mismatch_tmp], axis=1)
            img_tmp = ax[i, j].imshow(image_tmp, cmap='turbo')
            # plt.colorbar(mappable=img_tmp, ax=ax[i, j], fraction=0.015, pad=0.005)
            ax[i, j].set_title(f"obj: {ailoc.common.cpu(objstage_tmp[nidx[i_slice]]):.0f} nm",
                               fontsize=font_size)
            i_slice += 1
    figure.suptitle(f"data, model, error, RRSE={rrse:.2f}%")

    # plot the zernike coefficients
    figure, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 8))
    aberrations_names = []
    for i in range(psf_params_fitted['zernike_mode'].shape[0]):
        aberrations_names.append(f"{psf_params_fitted['zernike_mode'][i, 0]:.0f}, {psf_params_fitted['zernike_mode'][i, 1]:.0f}")
    plt.xticks(np.arange(psf_params_fitted['zernike_mode'].shape[0]),
               labels=aberrations_names, rotation=30, fontsize=font_size)
    bar_tmp = ax.bar(np.arange(psf_params_fitted['zernike_mode'].shape[0]), psf_params_fitted['zernike_coef'],
                     width=1, color='orange',
                     edgecolor='k')
    plt.yticks(fontsize=font_size)
    def autolabel(rects):
        y_axis_length = rects.datavalues.max()-rects.datavalues.min()
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+0.1, y_axis_length/100+height if height > 0 else height-y_axis_length/40, '%.1f' % height,
                     fontsize=font_size-2)

    autolabel(bar_tmp)
    ax.tick_params(axis='both',
                   direction='out'
                   )
    ax.set_ylabel('Zernike coefficients (nm)', fontsize=font_size)
    ax.set_xlabel('Zernike modes', fontsize=font_size)

    plt.show()

    return results


def beads_psf_calibrate(params_dict: dict, napari_plot=False):
    # load bead stack
    beads_file_name = params_dict['beads_file_name']
    beads_data = tifffile.imread(beads_file_name)

    psf_params_dict = params_dict['psf_params_dict']

    camera_params_dict = params_dict['camera_params_dict']

    calib_params_dict = params_dict['calib_params_dict']
    calib_params_dict['raw_images'] = beads_data
    calib_params_dict['camera_params_dict'] = camera_params_dict
    calib_params_dict['roi_size'] = psf_params_dict['psf_size']
    calib_params_dict['psf_params_dict'] = psf_params_dict

    # preprocess the bead stack
    calib_params_dict = ailoc.common.segment_local_max_beads(calib_params_dict)

    # fit the noised beads
    t0 = time.time()
    results = ailoc.common.zernike_calibrate_3d_beads_stack(calib_params_dict)
    print(f"the fitting time is: {time.time() - t0}s")

    # print the fitting results
    print(f"the relative root of squared error is: "
          f"{np.linalg.norm((results['data_fitted'] - results['data_measured']).flatten(), ord=2) / np.linalg.norm(results['data_measured'].flatten(), ord=2) * 100}%")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    for i in range(len(results['psf_params_fitted']['zernike_mode'])):
        print(f"{ailoc.common.cpu(results['psf_params_fitted']['zernike_mode'][i])}: "
              f"{ailoc.common.cpu(results['psf_params_fitted']['zernike_coef'][i])}")

    # save the calibration results
    save_path = os.path.dirname(beads_file_name) + '/' + os.path.basename(beads_file_name).split('.')[
        0] + '_calib_results.mat'
    sio.savemat(save_path, results)
    print(f"the calibration results are saved in: {save_path}")

    # using napari to visualize the calibration results
    if napari_plot:
        ailoc.common.cmpdata_napari(results['data_measured'], results['data_fitted'])

