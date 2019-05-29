import pandas as pd
import numpy as np
from itertools import product
import os
import cv2
import tifffile as tiff
from cycifsuite.get_data import read_synapse_file


def image_based_qc(imstorpath, metadata, plate_ids=None, wells=None):
    finished_pwf = []
    if plate_ids is None:
        plate_ids = list(range(1, 19))
    if wells is None:
        wells = product('CDE', ['04', '05', '06', '07', '08', '09', '10'])
    for plate in plate_ids:
        for i, j in wells:
            well = i + j
            for fld in ['fld' + str(i) for i in range(1, 10)]:
                pwf = '_'.join([str(plate), 'cell', well, fld])
                if pwf in finished_pwf:
                    continue
                print('Processing {}...'.format(pwf))
                path = imstorpath + 'plate' + str(plate) + '/analysis/'
                nuc_mask = path + '_'.join([well, fld, 'nucleiLM.tif'])
                cyto_mask = path + '_'.join([well, fld, 'cytoLM.tif'])
                ffc = path + '_'.join([well, fld, 'FFC.tif'])
                image = tiff.imread(ffc)
                img_mask = tiff.imread(nuc_mask)
                img_mask = img_mask.reshape(2048, 2048)
                num_nucs, cells = find_num_cells_in_img(image, img_mask, pwf)
                if len(cells) != metadata[metadata.pwf == pwf].shape[0]:
                    num_nucs, cells = zip(
                        *[(num_nucs[i], x) for i, x in enumerate(cells) if x in metadata.index])
                metadata.loc[cells, 'num_nuclei_in_mask'] = num_nucs
                finished_pwf.append(pwf)
    return metadata, finished_pwf


def find_num_cells_in_mask(mask_img, kernal_size=(9, 9)):
    mask_img = mask_img / np.max(mask_img)
    mask_img *= 255
    mask_img = mask_img.astype(np.uint8)
    mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
    ret, thresh = cv2.threshold(
        mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones(kernal_size, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    ret, markers = cv2.connectedComponents(opening)
    # background is also considered as a connected region
    n_cells = len(np.unique(markers)) - 1
    return n_cells


def find_num_cells_in_img(image, mask, pwf):
    list_nums_nuclei = []
    list_cells = []
    for mask_number in range(1, np.max(mask) + 1):
        loc_array = (mask == mask_number)
        cell_img_x, cell_img_y = np.nonzero(loc_array == True)
        cell_left, cell_right = cell_img_x.min(), cell_img_x.max()
        cell_bottom, cell_top = cell_img_y.min(), cell_img_y.max()
        cell_img = image[0, 3, 0, cell_left:cell_right +
                         1, cell_bottom:cell_top + 1]
        cell_img_mask = loc_array[
            cell_left:cell_right + 1, cell_bottom:cell_top + 1]
        cell_img = cell_img * cell_img_mask
        cell_name = pwf + '_' + str(mask_number)
        list_nums_nuclei.append(find_num_cells_in_mask(cell_img))
        list_cells.append(cell_name)
    return list_nums_nuclei, list_cells
