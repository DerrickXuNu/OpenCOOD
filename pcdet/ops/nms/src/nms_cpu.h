#ifndef NMS_CPU_H
#define NMS_CPU_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <boost/geometry.hpp>
template<typename DType, typename ShapeContainer>
inline py::array_t<DType> constant(ShapeContainer shape, DType value){
    // create ROWMAJOR array.
    py::array_t<DType> array(shape);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), value);
    return array;
}

template<typename DType>
inline py::array_t<DType> zeros(std::vector<long int> shape){
    return constant<DType, std::vector<long int>>(shape, 0);
}

template <typename DType>
std::vector<int> non_max_suppression_cpu(
    py::array_t<DType> boxes,
    py::array_t<int> order,
    DType thresh,
    DType eps=0)
{
    auto ndets = boxes.shape(0);
    auto boxes_r = boxes.template unchecked<2>();
    auto order_r = order.template unchecked<1>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto area = zeros<DType>({ndets});
    auto area_rw = area.template mutable_unchecked<1>();
    // get areas
    for(int i = 0; i < ndets; ++i){
        area_rw(i) = (boxes_r(i, 2) - boxes_r(i, 0) + eps) * (boxes_r(i, 3) - boxes_r(i, 1) + eps);
    }
    std::vector<int> keep;
    int i, j;
    DType xx1, xx2, w, h, inter, ovr;
    for(int _i = 0; _i < ndets; ++_i){
        i = order_r(_i);
        if(suppressed_rw(i) == 1)
            continue;
        keep.push_back(i);
        for(int _j = _i + 1; _j < ndets; ++_j){
            j = order_r(_j);
            if(suppressed_rw(j) == 1)
                continue;
            xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
            xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
            w = xx2 - xx1 + eps;
            if (w > 0){
                xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
                xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
                h = xx2 - xx1 + eps;
                if (h > 0){
                    inter = w * h;
                    ovr = inter / (area_rw(i) + area_rw(j) - inter);
                    if(ovr >= thresh)
                        suppressed_rw(j) = 1;
                }
            }
        }
    }
    return keep;
}

template <typename DType>
std::vector<int> rotate_non_max_suppression_cpu(
    py::array_t<DType> box_corners,
    py::array_t<int> order,
    py::array_t<DType> standup_iou,
    DType thresh)
{
    auto ndets = box_corners.shape(0);
    auto box_corners_r = box_corners.template unchecked<3>();
    auto order_r = order.template unchecked<1>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto standup_iou_r = standup_iou.template unchecked<2>();
    std::vector<int> keep;
    int i, j;

    namespace bg = boost::geometry;
    typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
    typedef bg::model::polygon<point_t> polygon_t;
    polygon_t poly, qpoly;
    std::vector<polygon_t> poly_inter, poly_union;
    DType inter_area, union_area, overlap;

    for(int _i = 0; _i < ndets; ++_i){
        i = order_r(_i);
        if(suppressed_rw(i) == 1)
            continue;
        keep.push_back(i);
        for(int _j = _i + 1; _j < ndets; ++_j){
            j = order_r(_j);
            if(suppressed_rw(j) == 1)
                continue;
            if (standup_iou_r(i, j) <= 0.0)
                continue;
            // std::cout << "pre_poly" << std::endl;
            try {
                bg::append(poly, point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                bg::append(poly, point_t(box_corners_r(i, 1, 0), box_corners_r(i, 1, 1)));
                bg::append(poly, point_t(box_corners_r(i, 2, 0), box_corners_r(i, 2, 1)));
                bg::append(poly, point_t(box_corners_r(i, 3, 0), box_corners_r(i, 3, 1)));
                bg::append(poly, point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 1, 0), box_corners_r(j, 1, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 2, 0), box_corners_r(j, 2, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 3, 0), box_corners_r(j, 3, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::intersection(poly, qpoly, poly_inter);
            } catch (const std::exception& e) {
                std::cout << "box i corners:" << std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i, k, 1) << std::endl;
                }
                std::cout << "box j corners:" <<  std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1) << std::endl;
                }
                // throw e;
                continue;
            }
            // std::cout << "post_poly" << std::endl;
            // std::cout << "post_intsec" << std::endl;
            if (!poly_inter.empty())
            {
                inter_area = bg::area(poly_inter.front());
                // std::cout << "pre_union" << " " << inter_area << std::endl;
                bg::union_(poly, qpoly, poly_union);
                /*
                if (poly_union.empty()){
                    std::cout << "intsec area:" << " " << inter_area << std::endl;
                    std::cout << "box i corners:" << std::endl;
                    for(int k = 0; k < 4; ++k){
                        std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i, k, 1) << std::endl;
                    }
                    std::cout << "box j corners:" <<  std::endl;
                    for(int k = 0; k < 4; ++k){
                        std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1) << std::endl;
                    }
                }*/
                // std::cout << "post_union" << poly_union.empty() << std::endl;
                if (!poly_union.empty()){ // ignore invalid box
                    union_area = bg::area(poly_union.front());
                    // std::cout << "post union area" << std::endl;
                    // std::cout << union_area << "debug" << std::endl;
                    overlap = inter_area / union_area;
                    if(overlap >= thresh)
                        suppressed_rw(j) = 1;
                    poly_union.clear();
                }
            }
            poly.clear();
            qpoly.clear();
            poly_inter.clear();

        }
    }
    return keep;
}




template <typename DType>
py::list IOU_weighted_rotate_non_max_suppression_cpu(
    py::array_t<DType> boxes,
    py::array_t<DType> box_corners,
    py::array_t<DType> standup_iou,
    DType thresh,
    DType cnt_thresh,
    py::array_t<DType> scores,
    py::array_t<DType> IOU_preds,
    py::array_t<int> labels,
    py::array_t<int> dirs,
    py::array_t<DType> anchors
    )
{
    // setting hyperparameters
    std::vector<DType> nms_sigma_dist_interval_r = {0, 20, 40, 60};
    std::vector<DType> nms_sigma_square_r = {0.0009, 0.009, 0.1, 1};
    DType suppressed_thresh = thresh;
    // end setting hyperparameters

    auto ndets = box_corners.shape(0);
    auto box_corners_r = box_corners.template unchecked<3>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto standup_iou_r = standup_iou.template unchecked<2>();
    auto scores_r = scores.template unchecked<1>();
    auto IOU_preds_r = IOU_preds.template unchecked<1>();
    auto boxes_r = boxes.template unchecked<2>();
    auto labels_r = labels.template unchecked<1>();
    auto dirs_r = dirs.template unchecked<1>();
    auto anchors_r = anchors.template unchecked<2>();

    namespace bg = boost::geometry;
    typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
    typedef bg::model::polygon<point_t> polygon_t;
    polygon_t poly, qpoly;
    std::vector<polygon_t> poly_inter, poly_union;
    DType inter_area, union_area, overlap;

    std::vector<std::vector<DType>> boxes_ret;
    std::vector<DType> scores_ret;
    std::vector<int> labels_ret;
    std::vector<int> dirs_ret;
    py::list ret;

    DType* scores_rw = (DType *)malloc(sizeof(DType) * ndets);
    for(int i = 0 ;i < ndets;i++){
        scores_rw[i] = scores_r(i);
    }
    DType weight_pos[7];
    DType avg_pos[7];

    std::vector<std::vector<DType>> box_ret;
    DType score_box;
    // centerness
    std::vector<DType> centerness;
    for(int i = 0 ;i < ndets;i++){
        //DType dist = sqrt(pow(boxes_r(i,0)-anchors_r(i,0),2)+pow(boxes_r(i,1)-anchors_r(i,1),2));
        DType dist = pow(boxes_r(i,0)-anchors_r(i,0),2)+pow(boxes_r(i,1)-anchors_r(i,1),2);  // possible a little bit better
        centerness.push_back(exp(dist));
    }
    DType centerness_sum = 0;
    for(int i = 0 ;i < ndets;i++){
        centerness_sum += centerness[i];
    }
    for(int i = 0 ;i < ndets;i++){
        centerness[i] /= centerness_sum;
        scores_rw[i] *= (1-centerness[i]);
    }

    // normalization for score here
    DType score_max4norm=-10000;
    for(int i = 0 ;i < ndets;i++){
        if(scores_rw[i] > score_max4norm){
	        score_max4norm = scores_rw[i];
	    }
    }
    for(int i = 0 ;i < ndets;i++){
        scores_rw[i] /= score_max4norm;
    }
    while(1){
        DType score_max = -1;
        int idx_max = -1;
        bool flag_all_checked = true;
        //find out the box with the maximum score
	    for(int i = 0; i < ndets; ++i){
            if(suppressed_rw(i) == 1)
                continue;
            flag_all_checked = false;
            if(scores_r(i) > score_max){
                score_max = scores_r(i);
                idx_max = i;
            }
        }
        if (flag_all_checked)
            break;
	    DType dist2origin = sqrt(pow(boxes_r(idx_max,0), 2) + pow(boxes_r(idx_max,1), 2));
        suppressed_rw(idx_max) = 1;
        for(int j = 0;j < 7;j++){
            weight_pos[j] = 0;
            avg_pos[j] = 0;
        }
        score_box = -1;
        DType cnt = 0;
        // if the box A fails to reach threshold, recover the boxes whose IOU with box A > 0.3
        std::vector<int> recover_list;
        // calculate the IOU
        for(int j = 0; j < ndets; ++j){
            try {
                bg::append(poly, point_t(box_corners_r(idx_max, 0, 0), box_corners_r(idx_max, 0, 1)));
                bg::append(poly, point_t(box_corners_r(idx_max, 1, 0), box_corners_r(idx_max, 1, 1)));
                bg::append(poly, point_t(box_corners_r(idx_max, 2, 0), box_corners_r(idx_max, 2, 1)));
                bg::append(poly, point_t(box_corners_r(idx_max, 3, 0), box_corners_r(idx_max, 3, 1)));
                bg::append(poly, point_t(box_corners_r(idx_max, 0, 0), box_corners_r(idx_max, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 1, 0), box_corners_r(j, 1, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 2, 0), box_corners_r(j, 2, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 3, 0), box_corners_r(j, 3, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::intersection(poly, qpoly, poly_inter);
            } catch (const std::exception& e) {
                std::cout << "box i corners:" << std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(idx_max, k, 0) << " " << box_corners_r(idx_max, k, 1) << std::endl;
                }
                std::cout << "box j corners:" <<  std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1) << std::endl;
                }
                continue;
            }

            if (!poly_inter.empty())
            {
                inter_area = bg::area(poly_inter.front());
                bg::union_(poly, qpoly, poly_union);
                if (!poly_union.empty()){ // ignore invalid box
                    union_area = bg::area(poly_union.front());
                    overlap = inter_area / union_area;

                    //counter for box[idx_max]
                    if(overlap > 0 && labels_r(j) == labels_r(idx_max)){
                        cnt += overlap * IOU_preds_r(j);
                    }
                    // weighted average bounding box, pick boxes whose IOU > 0.3 and with the same label
                    if(overlap > 0.3 && labels_r(j) == labels_r(idx_max)){
                        if (score_box < scores_rw[j]){
                            score_box = scores_rw[j];
                        }
                        DType IOU_weight = 0;

                        // the larger the IOU box B is with box[idx_max], the more weight will be put on B
                        // sigma increases as the dist2origin increases,
                        for(unsigned int k = 0 ;k < nms_sigma_dist_interval_r.size() - 1;k++){
                            DType dist_l = nms_sigma_dist_interval_r[k];
                            DType dist_r = nms_sigma_dist_interval_r[k+1];
                            if(dist2origin >= dist_l && dist2origin < dist_r){
                                IOU_weight = exp(-pow(1-overlap,2)/nms_sigma_square_r[k]);
                            }
                        }
                        for(int k = 0;k < 7;k++){
                            avg_pos[k] += IOU_weight * IOU_preds_r(j) * boxes_r(j,k);
                            weight_pos[k] += IOU_weight * IOU_preds_r(j);
                        }
                    }
                    // suppress the box whose IOU with box[idx_max] > suppressed_thresh
                    if (suppressed_rw(j) != 1 && standup_iou_r(idx_max, j) > 0){
                        if(overlap >= suppressed_thresh){
                            suppressed_rw(j) = 1;
                            recover_list.push_back(j);
                        }
                    }
                    poly_union.clear();
                }
            }
            poly.clear();
            qpoly.clear();
            poly_inter.clear();
        }

	// threshold set to 2.6
	if(cnt > cnt_thresh){
	    scores_ret.push_back(score_box * score_max4norm);
	    // calculate avg pos
	    for(int k = 0 ;k < 7;k++){
	        avg_pos[k] /= weight_pos[k];
	    }
	    std::vector<DType> box_ret;
        for(int k = 0;k < 7;k++){
            box_ret.push_back(avg_pos[k]);
        }
        boxes_ret.push_back(box_ret);
	    labels_ret.push_back(labels_r(idx_max));
	    dirs_ret.push_back(dirs_r(idx_max));
        }else{
        // recover the pred bboxes that has been suppressed
	    for(unsigned int k = 0 ;k < recover_list.size();k++){
	        suppressed_rw(recover_list[k]) = 0;
	    }
	}
    }
    ret.append(boxes_ret);
    ret.append(scores_ret);
    ret.append(labels_ret);
    ret.append(dirs_ret);
    return ret;
}


#endif
