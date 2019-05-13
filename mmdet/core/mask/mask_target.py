import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def mask_iou_target(pos_proposals_list, pos_assigned_gt_inds_list,
                    gt_mask_polys_list, mask_preds, mask_targets, img_metas):
    area_ratios = map(mask_iou_target_single, pos_proposals_list,
                      pos_assigned_gt_inds_list, gt_mask_polys_list, img_metas)
    area_ratios = torch.cat(list(area_ratios))
    assert mask_targets.size(0) == area_ratios.size(0)
    mask_pred = (mask_preds > 0.5).float()  # binarize mask pred
    mask_overlaps = (mask_pred * mask_targets).sum((-1, -2))
    full_poly_areas = mask_targets.sum((-1, -2)) / area_ratios
    mask_unions = mask_pred.sum((-1, -2)) + full_poly_areas - mask_overlaps
    mask_iou_targets = mask_overlaps / mask_unions
    return mask_iou_targets


def mask_iou_target_single(pos_proposals, pos_assigned_gt_inds, gt_mask_polys,
                           img_meta):
    num_pos = pos_proposals.size(0)
    h, w, _ = img_meta['img_shape']

    area_ratios = []
    if num_pos > 0:
        pos_proposals_numpy = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        pos_mask_polys = [
            gt_mask_polys[pos_assigned_gt_inds[i]] for i in range(num_pos)
        ]
        for proposal, mask_polys in zip(pos_proposals_numpy, pos_mask_polys):
            cropped_mask_polys = crop_polygons(mask_polys, proposal)
            x1 = int(proposal[0])
            y1 = int(proposal[1])
            x2 = int(proposal[2]) + 1
            y2 = int(proposal[3]) + 1
            for poly_ in mask_polys:
                poly = np.array(poly_, dtype=np.float32)
                x1 = np.minimum(x1, poly[0::2].min())
                x2 = np.maximum(x2, poly[0::2].max())
                y1 = np.minimum(y1, poly[1::2].min())
                y2 = np.maximum(y2, poly[1::2].max())
            x1 = np.maximum(x1, 0)
            x2 = np.minimum(x2, w - 1)
            y1 = np.maximum(y1, 0)
            y2 = np.minimum(y2, h - 1)

            full_polys = crop_polygons(mask_polys, [x1, y1, x2, y2])
            rle_of_full_polys = mask_util.frPyObjects(full_polys, y2 - y1,
                                                      x2 - x1)
            full_poly_area = mask_util.area(rle_of_full_polys).sum()

            rle_of_proposal_polys = mask_util.frPyObjects(
                cropped_mask_polys, proposal[3] - proposal[1],
                proposal[2] - proposal[0])
            proposal_poly_area = mask_util.area(rle_of_proposal_polys).sum()
            mask_ratio = proposal_poly_area / full_poly_area
            area_ratios.append(mask_ratio)
        area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
            pos_proposals.device) + 1e-7
    else:
        area_ratios = pos_proposals.new_zeros((0, ))
    return area_ratios


def crop_polygons(polygons, box):
    cropped_polygons = []
    for poly in polygons:
        p = poly.copy()
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]
        cropped_polygons.append(p)

    return cropped_polygons
