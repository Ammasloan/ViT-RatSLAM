/*
 * openRatSLAM
 *
 * utils - General purpose utility helper functions mainly for angles and
 * readings settings
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
#include "local_view_match.h"
#include "../utils/utils.h"

#include <assert.h>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string.h>

using namespace std;
#include <algorithm>
#include <boost/foreach.hpp>

#include <chrono> // Timing diagnostics
#include <stdio.h>


namespace ratslam {

LocalViewMatch::LocalViewMatch(ptree settings) {
  use_vpr_backend = false;
  get_setting_from_ptree(VT_MIN_PATCH_NORMALISATION_STD, settings,
                         "vt_min_patch_normalisation_std", (double)0);
  get_setting_from_ptree(VT_PATCH_NORMALISATION, settings, "vt_patch_normalise",
                         0);
  get_setting_from_ptree(VT_NORMALISATION, settings, "vt_normalisation",
                         (double)0);
  get_setting_from_ptree(VT_SHIFT_MATCH, settings, "vt_shift_match", 25);
  get_setting_from_ptree(VT_STEP_MATCH, settings, "vt_step_match", 5);
  get_setting_from_ptree(VT_PANORAMIC, settings, "vt_panoramic", 0);
  get_setting_from_ptree(vt_log, settings, "vt_log", false);
  vt_log_separator_every = settings.get<int>("vt_log_separator_every", 15);
  if (vt_log_separator_every < 0)
    vt_log_separator_every = 0;
  vt_log_event_count = 0;

  get_setting_from_ptree(VT_MATCH_THRESHOLD, settings, "vt_match_threshold",
                         0.03);
  get_setting_from_ptree(TEMPLATE_X_SIZE, settings, "template_x_size", 1);
  get_setting_from_ptree(TEMPLATE_Y_SIZE, settings, "template_y_size", 1);
  get_setting_from_ptree(IMAGE_VT_X_RANGE_MIN, settings, "image_crop_x_min", 0);
  get_setting_from_ptree(IMAGE_VT_X_RANGE_MAX, settings, "image_crop_x_max",
                         -1);
  get_setting_from_ptree(IMAGE_VT_Y_RANGE_MIN, settings, "image_crop_y_min", 0);
  get_setting_from_ptree(IMAGE_VT_Y_RANGE_MAX, settings, "image_crop_y_max",
                         -1);

  TEMPLATE_SIZE = TEMPLATE_X_SIZE * TEMPLATE_Y_SIZE;

  templates.reserve(10000);

  current_view.resize(TEMPLATE_SIZE);

  std::string backend_choice;
  get_setting_from_ptree(backend_choice, settings, "vpr_backend",
                         (std::string) "sad");
  std::transform(backend_choice.begin(), backend_choice.end(),
                 backend_choice.begin(), ::tolower);
  use_vpr_backend = (backend_choice == "salad" || backend_choice == "vit");

  current_vt = 0;
  prev_vt = 0;
}

LocalViewMatch::~LocalViewMatch() {}

void LocalViewMatch::on_image(const unsigned char *view_rgb, bool greyscale,
                              unsigned int image_width,
                              unsigned int image_height) {
  if (use_vpr_backend)
    return;

  if (view_rgb == NULL)
    return;

  // Timing diagnostics
  static int frame_count = 0;
  static double total_match_time_ms = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  IMAGE_WIDTH = image_width;
  IMAGE_HEIGHT = image_height;

  if (IMAGE_VT_X_RANGE_MAX == -1)
    IMAGE_VT_X_RANGE_MAX = IMAGE_WIDTH;
  if (IMAGE_VT_Y_RANGE_MAX == -1)
    IMAGE_VT_Y_RANGE_MAX = IMAGE_HEIGHT;

  this->view_rgb = view_rgb;
  this->greyscale = greyscale;

  convert_view_to_view_template(greyscale);
  prev_vt = get_current_vt();
  unsigned int vt_match_id;
  compare(vt_error, vt_match_id);

  // Measure time
  auto end_time = std::chrono::high_resolution_clock::now();
  double match_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  total_match_time_ms += match_time_ms;
  frame_count++;

  // Print timing every 100 frames
  if (frame_count % 100 == 0) {
    cout << "[TIMING] Frame " << frame_count
         << " | Templates: " << templates.size() << " | This: " << fixed
         << setprecision(1) << match_time_ms << "ms"
         << " | Avg: " << (total_match_time_ms / frame_count) << "ms" << endl;
  }

  bool any_log = false;
  if (vt_error <= VT_MATCH_THRESHOLD) {
    set_current_vt((int)vt_match_id);
    if (vt_log) {
      cout << "VTM[" << setw(4) << get_current_vt() << "] " << endl;
      any_log = true;
    }
  } else {
    vt_relative_rad = 0;
    set_current_vt(create_template());
    if (vt_log) {
      cout << "VTN[" << setw(4) << get_current_vt() << "] " << endl;
      any_log = true;
    }
  }

  vt_log_event_count++;
  if (vt_log_separator_every > 0 &&
      (vt_log_event_count %
       static_cast<unsigned int>(vt_log_separator_every)) == 0) {
    cout << "-------------------- [VT_EVENT] " << vt_log_event_count
         << " --------------------" << endl;
    any_log = true;
  }

  if (any_log)
    cout.flush();
}

void LocalViewMatch::on_vpr_embedding(const std::vector<float> &embedding,
                                      int template_id, bool is_new,
                                      float match_score) {
  if (!use_vpr_backend)
    return;

  prev_vt = get_current_vt();
  int vt_id = -1;
  bool created = false;

  std::unordered_map<int, int>::iterator it =
      vpr_template_map.find(template_id);
  if (is_new || it == vpr_template_map.end()) {
    vt_id = create_template_from_embedding(embedding);
    vpr_template_map[template_id] = vt_id;
    created = true;
    vt_error = 0.0;
  } else {
    vt_id = it->second;
    set_current_vt(vt_id);
    double score = match_score;
    if (score < 0.0)
      score = 0.0;
    if (score > 1.0)
      score = 1.0;
    vt_error = 1.0 - score;
  }

  vt_relative_rad = 0;

  if (vt_log) {
    if (created)
      cout << "VTN[" << setw(4) << get_current_vt() << "] " << endl;
    else
      cout << "VTM[" << setw(4) << get_current_vt() << "] " << endl;
    vt_log_event_count++;
    if (vt_log_separator_every > 0 &&
        (vt_log_event_count %
         static_cast<unsigned int>(vt_log_separator_every)) == 0) {
      cout << "-------------------- [VT_EVENT] " << vt_log_event_count
           << " --------------------" << endl;
    }
    cout.flush();
  }
}

void LocalViewMatch::clip_view_x_y(int &x, int &y) {
  if (x < 0)
    x = 0;
  else if (x > TEMPLATE_X_SIZE - 1)
    x = TEMPLATE_X_SIZE - 1;

  if (y < 0)
    y = 0;
  else if (y > TEMPLATE_Y_SIZE - 1)
    y = TEMPLATE_Y_SIZE - 1;
}

void LocalViewMatch::convert_view_to_view_template(bool grayscale) {
  int data_next = 0;
  int sub_range_x = IMAGE_VT_X_RANGE_MAX - IMAGE_VT_X_RANGE_MIN;
  int sub_range_y = IMAGE_VT_Y_RANGE_MAX - IMAGE_VT_Y_RANGE_MIN;
  int x_block_size = sub_range_x / TEMPLATE_X_SIZE;
  int y_block_size = sub_range_y / TEMPLATE_Y_SIZE;
  int pos;

  for (unsigned int i; i < current_view.size(); i++)
    current_view[i] = 0;

  if (grayscale) {
    for (int y_block = IMAGE_VT_Y_RANGE_MIN, y_block_count = 0;
         y_block_count < TEMPLATE_Y_SIZE;
         y_block += y_block_size, y_block_count++) {
      for (int x_block = IMAGE_VT_X_RANGE_MIN, x_block_count = 0;
           x_block_count < TEMPLATE_X_SIZE;
           x_block += x_block_size, x_block_count++) {
        for (int x = x_block; x < (x_block + x_block_size); x++) {
          for (int y = y_block; y < (y_block + y_block_size); y++) {
            pos = (x + y * IMAGE_WIDTH);
            current_view[data_next] += (double)(view_rgb[pos]);
          }
        }
        current_view[data_next] /= (255.0);
        current_view[data_next] /= (x_block_size * y_block_size);
        data_next++;
      }
    }
  } else {
    for (int y_block = IMAGE_VT_Y_RANGE_MIN, y_block_count = 0;
         y_block_count < TEMPLATE_Y_SIZE;
         y_block += y_block_size, y_block_count++) {
      for (int x_block = IMAGE_VT_X_RANGE_MIN, x_block_count = 0;
           x_block_count < TEMPLATE_X_SIZE;
           x_block += x_block_size, x_block_count++) {
        for (int x = x_block; x < (x_block + x_block_size); x++) {
          for (int y = y_block; y < (y_block + y_block_size); y++) {
            pos = (x + y * IMAGE_WIDTH) * 3;
            current_view[data_next] +=
                ((double)(view_rgb[pos]) + (double)(view_rgb[pos + 1]) +
                 (double)(view_rgb[pos + 2]));
          }
        }
        current_view[data_next] /= (255.0 * 3.0);
        current_view[data_next] /= (x_block_size * y_block_size);

        data_next++;
      }
    }
  }

  if (VT_NORMALISATION > 0) {
    double avg_value = 0;

    for (unsigned int i = 0; i < current_view.size(); i++) {
      avg_value += current_view[i];
    }

    avg_value /= current_view.size();

    for (unsigned int i = 0; i < current_view.size(); i++) {
      current_view[i] = std::max(
          0.0, std::min(current_view[i] * VT_NORMALISATION / avg_value, 1.0));
    }
  }

  // now do patch normalisation
  // +- patch size on the pixel, ie 4 will give a 9x9
  if (VT_PATCH_NORMALISATION > 0) {
    int patch_size = VT_PATCH_NORMALISATION;
    int patch_total = (patch_size * 2 + 1) * (patch_size * 2 + 1);
    double patch_sum;
    double patch_mean;
    double patch_std;
    int patch_x_clip;
    int patch_y_clip;

    // first make a copy of the view
    std::vector<double> current_view_copy;
    current_view_copy.resize(current_view.size());
    for (unsigned int i = 0; i < current_view.size(); i++)
      current_view_copy[i] = current_view[i];

    // this code could be significantly optimimised ....
    for (int x = 0; x < TEMPLATE_X_SIZE; x++) {
      for (int y = 0; y < TEMPLATE_Y_SIZE; y++) {
        patch_sum = 0;
        for (int patch_x = x - patch_size; patch_x < x + patch_size + 1;
             patch_x++) {
          for (int patch_y = y - patch_size; patch_y < y + patch_size + 1;
               patch_y++) {
            patch_x_clip = patch_x;
            patch_y_clip = patch_y;
            clip_view_x_y(patch_x_clip, patch_y_clip);

            patch_sum += current_view_copy[patch_x_clip +
                                           patch_y_clip * TEMPLATE_X_SIZE];
          }
        }
        patch_mean = patch_sum / patch_total;

        patch_sum = 0;
        for (int patch_x = x - patch_size; patch_x < x + patch_size + 1;
             patch_x++) {
          for (int patch_y = y - patch_size; patch_y < y + patch_size + 1;
               patch_y++) {
            patch_x_clip = patch_x;
            patch_y_clip = patch_y;
            clip_view_x_y(patch_x_clip, patch_y_clip);

            patch_sum += ((current_view_copy[patch_x_clip +
                                             patch_y_clip * TEMPLATE_X_SIZE] -
                           patch_mean) *
                          (current_view_copy[patch_x_clip +
                                             patch_y_clip * TEMPLATE_X_SIZE] -
                           patch_mean));
          }
        }

        patch_std = sqrt(patch_sum / patch_total);

        if (patch_std < VT_MIN_PATCH_NORMALISATION_STD)
          current_view[x + y * TEMPLATE_X_SIZE] = 0.5;
        else {
          current_view[x + y * TEMPLATE_X_SIZE] = max(
              (double)0,
              min(1.0,
                  (((current_view_copy[x + y * TEMPLATE_X_SIZE] - patch_mean) /
                    patch_std) +
                   3.0) /
                      6.0));
        }
      }
    }
  }

  double sum = 0;

  // find the mean of the data
  for (int i = 0; i < current_view.size(); i++)
    sum += current_view[i];

  current_mean = sum / current_view.size();
}

// create and add a visual template to the collection
int LocalViewMatch::create_template() {
  templates.resize(templates.size() + 1);
  VisualTemplate *new_template = &(*(templates.end() - 1));

  new_template->id = templates.size() - 1;
  double *data_ptr = &current_view[0];
  new_template->data.reserve(TEMPLATE_SIZE);
  for (int i = 0; i < TEMPLATE_SIZE; i++)
    new_template->data.push_back(*(data_ptr++));

  new_template->mean = current_mean;

  return templates.size() - 1;
}

int LocalViewMatch::create_template_from_embedding(
    const std::vector<float> &embedding) {
  templates.resize(templates.size() + 1);
  VisualTemplate *new_template = &(*(templates.end() - 1));
  new_template->id = templates.size() - 1;
  new_template->data.reserve(embedding.size());
  for (size_t i = 0; i < embedding.size(); ++i)
    new_template->data.push_back(static_cast<double>(embedding[i]));

  new_template->mean = 0.0;
  set_current_vt(new_template->id);
  return new_template->id;
}

// compare a visual template to all the stored templates, allowing for
// slen pixel shifts in each direction
// returns the matching template and the MSE
void LocalViewMatch::compare(double &vt_err, unsigned int &vt_match_id) {
  if (templates.size() == 0) {
    vt_err = DBL_MAX;
    vt_error = vt_err;
    return;
  }

  double *data = &current_view[0];
  double mindiff, cdiff;
  mindiff = DBL_MAX;

  vt_err = DBL_MAX;
  int min_template = 0;

  double *template_ptr;
  double *column_ptr;
  double *template_row_ptr;
  double *column_row_ptr;
  double *template_start_ptr;
  double *column_start_ptr;
  int row_size;
  int sub_row_size;
  double *column_end_ptr;
  VisualTemplate vt;
  int min_offset;

  int offset;
  double epsilon = 0.005;

  // Diagnostic counters
  int templates_compared = 0;
  int templates_skipped = 0;

  if (VT_PANORAMIC) {

    BOOST_FOREACH (vt, templates) {

      if (abs(current_mean - vt.mean) > VT_MATCH_THRESHOLD + epsilon) {
        templates_skipped++;
        continue;
      }
      templates_compared++;

      // for each vt try matching the view at different offsets
      // try to fast break based on error already great than previous errors
      // handles 2d images shifting only in the x direction
      // note I haven't tested on a 1d yet.
      for (offset = 0; offset < TEMPLATE_X_SIZE; offset += VT_STEP_MATCH) {
        cdiff = 0;
        template_start_ptr = &vt.data[0] + offset;
        column_start_ptr = &data[0];
        row_size = TEMPLATE_X_SIZE;
        column_end_ptr = &data[0] + TEMPLATE_SIZE - offset;
        sub_row_size = TEMPLATE_X_SIZE - offset;

        // do from offset to end
        for (column_row_ptr = column_start_ptr,
            template_row_ptr = template_start_ptr;
             column_row_ptr < column_end_ptr;
             column_row_ptr += row_size, template_row_ptr += row_size) {
          for (column_ptr = column_row_ptr, template_ptr = template_row_ptr;
               column_ptr < column_row_ptr + sub_row_size;
               column_ptr++, template_ptr++) {
            cdiff += abs(*column_ptr - *template_ptr);
          }

          // fast breaks
          if (cdiff > mindiff)
            break;
        }

        // do from start to offset
        template_start_ptr = &vt.data[0];
        column_start_ptr = &data[0] + TEMPLATE_X_SIZE - offset;
        row_size = TEMPLATE_X_SIZE;
        column_end_ptr = &data[0] + TEMPLATE_SIZE;
        sub_row_size = offset;
        for (column_row_ptr = column_start_ptr,
            template_row_ptr = template_start_ptr;
             column_row_ptr < column_end_ptr;
             column_row_ptr += row_size, template_row_ptr += row_size) {
          for (column_ptr = column_row_ptr, template_ptr = template_row_ptr;
               column_ptr < column_row_ptr + sub_row_size;
               column_ptr++, template_ptr++) {
            cdiff += abs(*column_ptr - *template_ptr);
          }

          // fast breaks
          if (cdiff > mindiff)
            break;
        }

        if (cdiff < mindiff) {
          mindiff = cdiff;
          min_template = vt.id;
          min_offset = offset;
        }
      }
    }

    vt_relative_rad = (double)min_offset / TEMPLATE_X_SIZE * 2.0 * M_PI;
    if (vt_relative_rad > M_PI)
      vt_relative_rad = vt_relative_rad - 2.0 * M_PI;
    vt_err = mindiff / (double)TEMPLATE_SIZE;
    vt_match_id = min_template;

    vt_error = vt_err;

  } else {

    BOOST_FOREACH (vt, templates) {

      if (abs(current_mean - vt.mean) > VT_MATCH_THRESHOLD + epsilon) {
        templates_skipped++;
        continue;
      }
      templates_compared++;

      // for each vt try matching the view at different offsets
      // try to fast break based on error already great than previous errors
      // handles 2d images shifting only in the x direction
      // note I haven't tested on a 1d yet.
      for (offset = 0; offset < VT_SHIFT_MATCH * 2 + 1;
           offset += VT_STEP_MATCH) {
        cdiff = 0;
        template_start_ptr = &vt.data[0] + offset;
        column_start_ptr = &data[0] + VT_SHIFT_MATCH;
        row_size = TEMPLATE_X_SIZE;
        column_end_ptr = &data[0] + TEMPLATE_SIZE - VT_SHIFT_MATCH;
        sub_row_size = TEMPLATE_X_SIZE - 2 * VT_SHIFT_MATCH;

        for (column_row_ptr = column_start_ptr,
            template_row_ptr = template_start_ptr;
             column_row_ptr < column_end_ptr;
             column_row_ptr += row_size, template_row_ptr += row_size) {
          for (column_ptr = column_row_ptr, template_ptr = template_row_ptr;
               column_ptr < column_row_ptr + sub_row_size;
               column_ptr++, template_ptr++) {
            cdiff += abs(*column_ptr - *template_ptr);
          }

          // fast breaks
          if (cdiff > mindiff)
            break;
        }

        if (cdiff < mindiff) {
          mindiff = cdiff;
          min_template = vt.id;
          min_offset = 0;
        }
      }
    }

    vt_relative_rad = 0;
    vt_err = mindiff /
             (double)(TEMPLATE_SIZE - 2 * VT_SHIFT_MATCH * TEMPLATE_Y_SIZE);
    vt_match_id = min_template;

    vt_error = vt_err;
  }

  // Log comparison stats (every 50 frames or on match)
  static int call_count = 0;
  call_count++;
  bool matched = (vt_err <= VT_MATCH_THRESHOLD);
  if (call_count % 50 == 0 || matched) {
    cout << "[COMPARE] #" << call_count << " | Total:" << templates.size()
         << " Compared:" << templates_compared
         << " Skipped:" << templates_skipped << " | Err:" << fixed
         << setprecision(4) << vt_err << " Thresh:" << VT_MATCH_THRESHOLD;
    if (matched)
      cout << " | MATCH->" << vt_match_id;
    else
      cout << " | NO_MATCH";
    cout << endl;
  }
}

} // namespace ratslam
