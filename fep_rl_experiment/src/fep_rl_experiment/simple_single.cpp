/*****************************
 Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are
 permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list
 of conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
 WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those of the
 authors and should not be interpreted as representing official policies, either expressed
 or implied, of Rafael Muñoz Salinas.
 ********************************/
/**
 * @file simple_single.cpp
 * @author Bence Magyar
 * @date June 2012
 * @version 0.1
 * @brief ROS version of the example named "simple" in the ArUco software package.
 */

#include <iostream>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <aruco_ros/aruco_ros_utils.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>

#include <dynamic_reconfigure/server.h>
#include <aruco_ros/ArucoThresholdConfig.h>

class ArucoSimple
{
private:
  cv::Mat inImage;
  aruco::CameraParameters camParam;
  tf::StampedTransform rightToLeft;
  bool useRectifiedImages;
  aruco::MarkerDetector mDetector;
  std::vector<aruco::Marker> markers;
  ros::Subscriber cam_info_sub;
  bool cam_info_received;
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  ros::Publisher pose_pub;
  ros::Publisher transform_pub;
  ros::Publisher position_pub;
  ros::Publisher marker_pub; // rviz visualization marker
  ros::Publisher pixel_pub;
  std::string marker_frame;
  std::string camera_frame;
  std::string reference_frame;

  double marker_size;
  double cube_size;
  std::vector<int> marker_ids;
  std::vector<tf::Transform> marker_offsets;

  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber image_sub;

  tf::TransformListener _tfListener;

  dynamic_reconfigure::Server<aruco_ros::ArucoThresholdConfig> dyn_rec_server;

public:
  ArucoSimple() :
      cam_info_received(false), nh("~"), it(nh)
  {

    if (nh.hasParam("corner_refinement"))
      ROS_WARN(
          "Corner refinement options have been removed in ArUco 3.0.0, corner_refinement ROS parameter is deprecated");

    aruco::MarkerDetector::Params params = mDetector.getParameters();
    std::string thresh_method;
    switch (params.thresMethod)
    {
      case aruco::MarkerDetector::ThresMethod::THRES_ADAPTIVE:
        thresh_method = "THRESH_ADAPTIVE";
        break;
      case aruco::MarkerDetector::ThresMethod::THRES_AUTO_FIXED:
        thresh_method = "THRESH_AUTO_FIXED";
        break;
      default:
        thresh_method = "UNKNOWN";
        break;
    }

    // Print parameters of ArUco marker detector:
    ROS_INFO_STREAM("Threshold method: " << thresh_method);

    float min_marker_size; // percentage of image area
    nh.param<float>("min_marker_size", min_marker_size, 0.02);

    std::string detection_mode;
    nh.param<std::string>("detection_mode", detection_mode, "DM_FAST");
    if (detection_mode == "DM_FAST")
      mDetector.setDetectionMode(aruco::DM_FAST, min_marker_size);
    else if (detection_mode == "DM_VIDEO_FAST")
      mDetector.setDetectionMode(aruco::DM_VIDEO_FAST, min_marker_size);
    else
      // Aruco version 2 mode
      mDetector.setDetectionMode(aruco::DM_NORMAL, min_marker_size);

    ROS_INFO_STREAM("Marker size min: " << min_marker_size << "% of image area");
    ROS_INFO_STREAM("Detection mode: " << detection_mode);

    image_sub = it.subscribe("/image", 1, &ArucoSimple::image_callback, this);
    cam_info_sub = nh.subscribe("/camera_info", 1, &ArucoSimple::cam_info_callback, this);

    image_pub = it.advertise("result", 1);
    debug_pub = it.advertise("debug", 1);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);
    transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
    position_pub = nh.advertise<geometry_msgs::Vector3Stamped>("position", 100);
    marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 10);
    pixel_pub = nh.advertise<geometry_msgs::PointStamped>("pixel", 10);

    nh.param<double>("marker_size", marker_size, 0.05);
    nh.param<double>("cube_size", cube_size, 0.05);
    nh.param<std::string>("reference_frame", reference_frame, "");
    nh.param<std::string>("camera_frame", camera_frame, "");
    nh.param<std::string>("marker_frame", marker_frame, "");
    nh.param<bool>("image_is_rectified", useRectifiedImages, true);

    std::string marker_ids_str;
    nh.param<std::string>("marker_ids", marker_ids_str, "500");
    // Convert the space-separated string into a vector of integers
    std::stringstream ss(marker_ids_str);
    int id;
    while (ss >> id) {
      marker_ids.push_back(id);
      if (ss.peek() == ' ') {
        ss.ignore();
      }
    }

    tf::Transform transform_1;
    transform_1.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_1.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));
    marker_offsets.push_back(transform_1);
    tf::Transform transform_2;
    transform_2.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_2.setRotation(tf::Quaternion(0.0, -0.7071, 0.0, 0.7071));
    marker_offsets.push_back(transform_2);
    tf::Transform transform_3;
    transform_3.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_3.setRotation(tf::Quaternion(0.0, -1.0, 0.0, 0.0));
    marker_offsets.push_back(transform_3);
    tf::Transform transform_4;
    transform_4.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_4.setRotation(tf::Quaternion(0.0, -0.7071, 0.0, -0.7071));
    marker_offsets.push_back(transform_4);
    tf::Transform transform_5;
    transform_5.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_5.setRotation(tf::Quaternion(0.7071, 0.0, 0.0, 0.7071));
    marker_offsets.push_back(transform_5);
    tf::Transform transform_6;
    transform_6.setOrigin(tf::Vector3(0.0, 0.0, -cube_size / 2.0));
    transform_6.setRotation(tf::Quaternion(-0.7071, 0.0, 0.0, 0.7071));
    marker_offsets.push_back(transform_6);

    ROS_ASSERT(camera_frame != "" && marker_frame != "");

    if (reference_frame.empty())
      reference_frame = camera_frame;

    ROS_INFO("ArUco node started with marker size of %f m and marker ids to track: [%s]", marker_size, ss.str().c_str());
    ROS_INFO("ArUco node will publish pose to TF with %s as parent and %s as child.", reference_frame.c_str(),
             marker_frame.c_str());

    dyn_rec_server.setCallback(boost::bind(&ArucoSimple::reconf_callback, this, _1, _2));
  }

  bool getTransform(const std::string& refFrame, const std::string& childFrame, tf::StampedTransform& transform)
  {
    std::string errMsg;

    if (!_tfListener.waitForTransform(refFrame, childFrame, ros::Time(0), ros::Duration(0.5), ros::Duration(0.01),
                                      &errMsg))
    {
      ROS_ERROR_STREAM("Unable to get pose from TF: " << errMsg);
      return false;
    }
    else
    {
      try
      {
        _tfListener.lookupTransform(refFrame, childFrame, ros::Time(0), // get latest available
                                    transform);
      }
      catch (const tf::TransformException& e)
      {
        ROS_ERROR_STREAM("Error in lookupTransform of " << childFrame << " in " << refFrame);
        return false;
      }

    }
    return true;
  }

  void image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
    if ((image_pub.getNumSubscribers() == 0) && (debug_pub.getNumSubscribers() == 0)
        && (pose_pub.getNumSubscribers() == 0) && (transform_pub.getNumSubscribers() == 0)
        && (position_pub.getNumSubscribers() == 0) && (marker_pub.getNumSubscribers() == 0)
        && (pixel_pub.getNumSubscribers() == 0))
    {
      ROS_DEBUG("No subscribers, not looking for ArUco markers");
      return;
    }

    static tf::TransformBroadcaster br;
    if (cam_info_received)
    {
      ros::Time curr_stamp = msg->header.stamp;
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        inImage = cv_ptr->image;

        // detection results will go into "markers"
        markers.clear();
        // ok, let's detect
        mDetector.detect(inImage, markers, camParam, marker_size, false);
        // for each marker, draw info and its boundaries in the image
        for (std::size_t i = 0; i < markers.size(); ++i)
        {
          // Check if the marker is in the list of marker_ids
          auto it = std::find(marker_ids.begin(), marker_ids.end(), markers[i].id);
          if (it != marker_ids.end())
          {
            int marker_index = std::distance(marker_ids.begin(), it);
            tf::Transform transform = aruco_ros::arucoMarker2Tf(markers[i]);
            tf::StampedTransform cameraToReference;
            cameraToReference.setIdentity();

            if (reference_frame != camera_frame)
            {
              getTransform(reference_frame, camera_frame, cameraToReference);
            }

            transform = static_cast<tf::Transform>(cameraToReference) * static_cast<tf::Transform>(rightToLeft)
                * transform;

            tf::StampedTransform stampedTransform(transform, curr_stamp, reference_frame, marker_frame);
            br.sendTransform(stampedTransform);

            tf::StampedTransform stampedCubeTransform(marker_offsets[marker_index], curr_stamp, marker_frame, "aruco_cube_frame");
            br.sendTransform(stampedCubeTransform);

            geometry_msgs::PoseStamped poseMsg;
            tf::poseTFToMsg(transform, poseMsg.pose);
            poseMsg.header.frame_id = reference_frame;
            poseMsg.header.stamp = curr_stamp;
            pose_pub.publish(poseMsg);

            geometry_msgs::TransformStamped transformMsg;
            tf::transformStampedTFToMsg(stampedTransform, transformMsg);
            transform_pub.publish(transformMsg);

            geometry_msgs::Vector3Stamped positionMsg;
            positionMsg.header = transformMsg.header;
            positionMsg.vector = transformMsg.transform.translation;
            position_pub.publish(positionMsg);

            geometry_msgs::PointStamped pixelMsg;
            pixelMsg.header = transformMsg.header;
            pixelMsg.point.x = markers[i].getCenter().x;
            pixelMsg.point.y = markers[i].getCenter().y;
            pixelMsg.point.z = 0;
            pixel_pub.publish(pixelMsg);

            // publish rviz marker representing the ArUco marker patch
            visualization_msgs::Marker visMarker;
            visMarker.header = transformMsg.header;
            visMarker.id = markers[i].id;
            visMarker.type = visualization_msgs::Marker::CUBE;
            visMarker.action = visualization_msgs::Marker::ADD;
            visMarker.pose = poseMsg.pose;
            visMarker.scale.x = marker_size;
            visMarker.scale.y = marker_size;
            visMarker.scale.z = 0.001;
            visMarker.color.r = 1.0;
            visMarker.color.g = 0;
            visMarker.color.b = 0;
            visMarker.color.a = 1.0;
            visMarker.lifetime = ros::Duration(3.0);
            marker_pub.publish(visMarker);
            break; // only publish the first marker found

          }
          // but drawing all the detected markers
          markers[i].draw(inImage, cv::Scalar(0, 0, 255), 2);
        }

        // draw a 3d cube in each marker if there is 3d info
        if (camParam.isValid() && marker_size != -1)
        {
          for (std::size_t i = 0; i < markers.size(); ++i)
          {
            aruco::CvDrawingUtils::draw3dAxis(inImage, markers[i], camParam);
          }
        }

        if (image_pub.getNumSubscribers() > 0)
        {
          // show input with augmented information
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = curr_stamp;
          out_msg.encoding = sensor_msgs::image_encodings::RGB8;
          out_msg.image = inImage;
          image_pub.publish(out_msg.toImageMsg());
        }

        if (debug_pub.getNumSubscribers() > 0)
        {
          // show also the internal image resulting from the threshold operation
          cv_bridge::CvImage debug_msg;
          debug_msg.header.stamp = curr_stamp;
          debug_msg.encoding = sensor_msgs::image_encodings::MONO8;
          debug_msg.image = mDetector.getThresholdedImage();
          debug_pub.publish(debug_msg.toImageMsg());
        }
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
  }

  // wait for one camerainfo, then shut down that subscriber
  void cam_info_callback(const sensor_msgs::CameraInfo &msg)
  {
    camParam = aruco_ros::rosCameraInfo2ArucoCamParams(msg, useRectifiedImages);

    // handle cartesian offset between stereo pairs
    // see the sensor_msgs/CameraInfo documentation for details
    rightToLeft.setIdentity();
    rightToLeft.setOrigin(tf::Vector3(-msg.P[3] / msg.P[0], -msg.P[7] / msg.P[5], 0.0));

    cam_info_received = true;
    cam_info_sub.shutdown();
  }

  void reconf_callback(aruco_ros::ArucoThresholdConfig &config, uint32_t level)
  {
    mDetector.setDetectionMode(aruco::DetectionMode(config.detection_mode), config.min_image_size);
    if (config.normalizeImage)
    {
      ROS_WARN("normalizeImageIllumination is unimplemented!");
    }
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "aruco_simple");

  ArucoSimple node;

  ros::spin();
}
