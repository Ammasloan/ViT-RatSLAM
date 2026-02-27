/*
 * openRatSLAM
 *
 * main_lv - ROS interface bindings for the local view cells
 *
 * Copyright (C) 2012
 * David Ball (david.ball@qut.edu.au) (1), Scott Heath
 * (scott.heath@uqconnect.edu.au) (2)
 *
 * RatSLAM algorithm by:
 * Michael Milford (1) and Gordon Wyeth (1) ([michael.milford,
 * gordon.wyeth]@qut.edu.au)
 *
 * 1. Queensland University of Technology, Australia
 * 2. The University of Queensland, Australia
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
using namespace std;

#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/utils.h"

#include <boost/property_tree/ini_parser.hpp>

#include <ratslam_ros/ViewTemplate.h>
#include <nav_msgs/Odometry.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>

#include <boost/function.hpp>
#include <image_transport/image_transport.h>

#include "ratslam/local_view_match.h"
#include <salad_vpr/Embedding.h>

#if HAVE_IRRLICHT
#include "graphics/local_view_scene.h"
ratslam::LocalViewScene *lvs = NULL;
bool use_graphics;
#endif

using namespace ratslam;
ratslam::LocalViewMatch *lv = NULL;
ros::Subscriber embedding_sub;
ros::Subscriber odom_sub;

static double wrap_pi(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

struct OdomYawSample {
  double t;
  double yaw;
};

class OdomYawTracker {
public:
  void configure(size_t max_samples, double max_dt) {
    max_samples_ = max_samples > 10 ? max_samples : 10;
    max_dt_ = max_dt > 0.0 ? max_dt : 0.5;
    trim();
  }

  void on_odom(const nav_msgs::OdometryConstPtr &odo) {
    double t = odo->header.stamp.toSec();
    if (t <= 0.0)
      t = ros::Time::now().toSec();

    if (!has_last_) {
      last_t_ = t;
      last_yaw_ = 0.0;
      has_last_ = true;
    } else {
      const double dt = t - last_t_;
      if (dt > 0.0 && dt < 10.0) {
        last_yaw_ = wrap_pi(last_yaw_ + odo->twist.twist.angular.z * dt);
      }
      last_t_ = t;
    }

    samples_.push_back({t, last_yaw_});
    trim();
  }

  bool lookup(const ros::Time &stamp, double *yaw_out) const {
    if (!yaw_out || samples_.empty())
      return false;

    double t = stamp.toSec();
    if (t <= 0.0)
      t = ros::Time::now().toSec();

    double best_dt = 1e9;
    double best_yaw = 0.0;
    for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) {
      const double dt = std::abs(it->t - t);
      if (dt < best_dt) {
        best_dt = dt;
        best_yaw = it->yaw;
        if (best_dt == 0.0)
          break;
      }
    }

    if (best_dt > max_dt_)
      return false;
    *yaw_out = best_yaw;
    return true;
  }

private:
  void trim() {
    while (samples_.size() > max_samples_)
      samples_.pop_front();
  }

  std::deque<OdomYawSample> samples_;
  size_t max_samples_ = 2000;
  double max_dt_ = 0.5;
  double last_t_ = 0.0;
  double last_yaw_ = 0.0;
  bool has_last_ = false;
};

static OdomYawTracker odom_yaw;
static std::unordered_map<unsigned int, double> vt_yaw_at_create;
static bool use_vpr_relative_rad = false;

void odom_callback(nav_msgs::OdometryConstPtr odo) { odom_yaw.on_odom(odo); }

void image_callback(sensor_msgs::ImageConstPtr image, ros::Publisher *pub_vt) {
  ROS_DEBUG_STREAM("LV:image_callback{" << ros::Time::now()
                                        << "} seq=" << image->header.seq);
  static bool encoding_printed = false;
  if (!encoding_printed) {
    ROS_INFO_STREAM("Image encoding: " << image->encoding);
    encoding_printed = true;
  }

  static ratslam_ros::ViewTemplate vt_output;

  // Treat rgb8/bgr8 as color; other encodings as greyscale.
  const bool is_bgr = (image->encoding == "bgr8");
  const bool is_rgb = (image->encoding == "rgb8");
  const bool is_color = (is_bgr || is_rgb);
  const bool is_greyscale = !is_color;

  if (image->data.empty())
    return;

  const unsigned char *data_ptr = image->data.data();

  // LocalViewScene draws color images using GL_BGR. If the decoded image is
  // rgb8, swap to BGR so the window colors are correct.
  static std::vector<unsigned char> bgr_buffer;
  if (is_color) {
    const size_t width = static_cast<size_t>(image->width);
    const size_t height = static_cast<size_t>(image->height);
    const size_t in_stride = static_cast<size_t>(image->step);
    const size_t out_stride = width * 3;
    const size_t required_in = in_stride * height;

    if (in_stride >= out_stride && image->data.size() >= required_in) {
      if (is_rgb || in_stride != out_stride) {
        const size_t out_size = out_stride * height;
        if (bgr_buffer.size() != out_size)
          bgr_buffer.resize(out_size);

        for (size_t y = 0; y < height; ++y) {
          const unsigned char *in_row = image->data.data() + y * in_stride;
          unsigned char *out_row = bgr_buffer.data() + y * out_stride;
          for (size_t x = 0; x < width; ++x) {
            const size_t idx = x * 3;
            if (is_rgb) {
              out_row[idx + 0] = in_row[idx + 2];
              out_row[idx + 1] = in_row[idx + 1];
              out_row[idx + 2] = in_row[idx + 0];
            } else {
              out_row[idx + 0] = in_row[idx + 0];
              out_row[idx + 1] = in_row[idx + 1];
              out_row[idx + 2] = in_row[idx + 2];
            }
          }
        }
        data_ptr = bgr_buffer.data();
      }
    }
  }

  lv->on_image(data_ptr, is_greyscale, image->width, image->height);

  vt_output.header.stamp = ros::Time::now();
  vt_output.header.seq++;
  vt_output.current_id = lv->get_current_vt();
  vt_output.relative_rad = lv->get_relative_rad();

  pub_vt->publish(vt_output);

#ifdef HAVE_IRRLICHT
  if (use_graphics) {
    lvs->draw_all();
  }
#endif
}

void embedding_callback(const salad_vpr::Embedding::ConstPtr &msg,
                        ros::Publisher *pub_vt) {
  if (!lv)
    return;

  std::vector<float> embedding(msg->data.begin(), msg->data.end());
  lv->on_vpr_embedding(embedding, msg->template_id, msg->is_new,
                       msg->match_score);

  static ratslam_ros::ViewTemplate vt_output;
  vt_output.header = msg->header;
  vt_output.header.seq++;
  vt_output.current_id = lv->get_current_vt();
  double relative_rad = lv->get_relative_rad();
  if (use_vpr_relative_rad) {
    double yaw_now = 0.0;
    if (odom_yaw.lookup(msg->header.stamp, &yaw_now)) {
      if (msg->is_new) {
        vt_yaw_at_create[vt_output.current_id] = yaw_now;
      } else {
        auto it = vt_yaw_at_create.find(vt_output.current_id);
        if (it != vt_yaw_at_create.end()) {
          relative_rad = wrap_pi(yaw_now - it->second);
        } else {
          vt_yaw_at_create[vt_output.current_id] = yaw_now;
        }
      }
    }
  }
  vt_output.relative_rad = relative_rad;
  pub_vt->publish(vt_output);

#ifdef HAVE_IRRLICHT
  if (use_graphics) {
    lvs->draw_all();
  }
#endif
}

int main(int argc, char *argv[]) {
  ROS_INFO_STREAM(
      argv[0]
      << " - openRatSLAM Copyright (C) 2012 David Ball and Scott Heath");
  ROS_INFO_STREAM("RatSLAM algorithm by Michael Milford and Gordon Wyeth");
  ROS_INFO_STREAM(
      "Distributed under the GNU GPL v3, see the included license file.");

  if (argc < 2) {
    ROS_FATAL_STREAM("USAGE: " << argv[0] << " <config_file>");
    exit(-1);
  }
  std::string topic_root = "";

  boost::property_tree::ptree settings, ratslam_settings, general_settings;
  read_ini(argv[1], settings);

  get_setting_child(general_settings, settings, "general", true);
  get_setting_from_ptree(topic_root, general_settings, "topic_root",
                         (std::string) "");
  get_setting_child(ratslam_settings, settings, "ratslam", true);
  lv = new ratslam::LocalViewMatch(ratslam_settings);

  // VPR relative_rad injection control (off by default; enable explicitly).
  // Default behavior: keep legacy SAD semantics (non-panoramic -> 0).
  std::string backend_choice;
  get_setting_from_ptree(backend_choice, ratslam_settings, "vpr_backend",
                         (std::string) "sad");
  std::transform(backend_choice.begin(), backend_choice.end(),
                 backend_choice.begin(), ::tolower);
  const bool is_vpr_backend = (backend_choice == "salad" || backend_choice == "vit");
  get_setting_from_ptree(use_vpr_relative_rad, ratslam_settings,
                         "vpr_use_relative_rad", false);

  int odom_hist = 2000;
  double odom_max_dt = 0.5;
  std::string vpr_odom_topic = topic_root + "/odom";
  get_setting_from_ptree(odom_hist, ratslam_settings,
                         "vpr_odom_history_size", odom_hist);
  get_setting_from_ptree(odom_max_dt, ratslam_settings,
                         "vpr_odom_max_dt", odom_max_dt);
  get_setting_from_ptree(vpr_odom_topic, ratslam_settings,
                         "vpr_odom_topic", vpr_odom_topic);
  std::string vpr_embedding_topic = topic_root + "/salad_vpr/embedding";
  get_setting_from_ptree(vpr_embedding_topic, ratslam_settings,
                         "vpr_embedding_topic", vpr_embedding_topic);

  if (!vpr_embedding_topic.empty() && vpr_embedding_topic[0] != '/' &&
      !topic_root.empty()) {
    if (topic_root.back() == '/')
      vpr_embedding_topic = topic_root + vpr_embedding_topic;
    else
      vpr_embedding_topic = topic_root + "/" + vpr_embedding_topic;
  }

  if (!ros::isInitialized()) {
    ros::init(argc, argv, "RatSLAMViewTemplate");
  }
  ros::NodeHandle node;

  ros::Publisher pub_vt = node.advertise<ratslam_ros::ViewTemplate>(
      topic_root + "/LocalView/Template", 0);

  if (use_vpr_relative_rad) {
    if (!is_vpr_backend) {
      ROS_WARN_STREAM(
          "vpr_use_relative_rad=1 but vpr_backend is not a VPR backend "
          "(sad); ignoring.");
      use_vpr_relative_rad = false;
    } else {
      odom_yaw.configure(static_cast<size_t>(odom_hist), odom_max_dt);
      odom_sub = node.subscribe<nav_msgs::Odometry>(
          vpr_odom_topic, 50, odom_callback, ros::VoidConstPtr(),
          ros::TransportHints().tcpNoDelay());
      ROS_INFO_STREAM("VPR relative_rad enabled (odom-integrated): topic="
                      << vpr_odom_topic << " max_dt=" << odom_max_dt
                      << " hist=" << odom_hist);
    }
  }

  image_transport::ImageTransport it(node);
  image_transport::Subscriber sub;

  if (lv->uses_vpr_backend()) {
    boost::function<void(const salad_vpr::Embedding::ConstPtr &)> cb =
        boost::bind(embedding_callback, _1, &pub_vt);
    embedding_sub = node.subscribe(vpr_embedding_topic, 10, cb);
    ROS_INFO_STREAM("VPR backend enabled, subscribing to embedding topic: "
                    << vpr_embedding_topic);
  } else {
    sub = it.subscribe(topic_root + "/camera/image", 0,
                       boost::bind(image_callback, _1, &pub_vt));
  }

#ifdef HAVE_IRRLICHT
  boost::property_tree::ptree draw_settings;
  get_setting_child(draw_settings, settings, "draw", true);
  get_setting_from_ptree(use_graphics, draw_settings, "enable", true);
  if (use_graphics)
    lvs = new ratslam::LocalViewScene(draw_settings, lv);
#endif

  ros::spin();

  return 0;
}
