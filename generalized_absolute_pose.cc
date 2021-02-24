// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#include <fstream>
#include <iostream>

#include "colmap/base/camera.h"
#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/util/random.h"
#include "colmap/optim/ransac.h"

using namespace colmap;

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::dict generalized_absolute_pose_estimation(
    const std::vector<Eigen::Vector2d> points2D,
    const std::vector<Eigen::Vector3d> points3D,
    const std::vector<Eigen::Matrix3x4d> camera_extrinsic,
    double max_error,
    double min_inlier_ratio,
    double confidence,
    double dyn_num_trials_multiplier,
    double min_num_trials,
    double max_num_trials) {
  // Build image information with points and extrinsics
  std::vector<GP3PEstimator::X_t> pts2d;
  for (int i = 0; i < points2D.size(); ++i) {
    pts2d.emplace_back();
    pts2d.back().rel_tform = camera_extrinsic[i];
    pts2d.back().xy = points2D[i];
  }

  // Build Ransac GPNP options
  RANSACOptions options;
  options.max_error = max_error;
  options.min_inlier_ratio = min_inlier_ratio;
  options.confidence = confidence;
  options.dyn_num_trials_multiplier = dyn_num_trials_multiplier;
  options.min_num_trials = min_num_trials;
  options.max_num_trials = max_num_trials;
  options.Check();

  // Generalized absolute pose estimation
  RANSAC<GP3PEstimator> ransac(options);

  // TODO Absolute pose refinement
  py::dict success_dict;
  auto report = ransac.Estimate(pts2d, points3D);
  success_dict["model"] = report.model;
  success_dict["success"] = report.success;
  success_dict["num_trials"] = report.num_trials;
  success_dict["inlier_mask"] = report.inlier_mask;
  return success_dict;
}
