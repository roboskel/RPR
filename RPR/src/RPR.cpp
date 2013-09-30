 #include <ros/ros.h>
    #include <sensor_msgs/PointCloud2.h>
    #include <pcl/ros/conversions.h>
    #include <pcl/point_types.h>
 	#include <pcl/point_cloud.h>
    #include <tf/transform_listener.h>
    #include <Eigen/Core>
    #include <boost/algorithm/string.hpp>
    #include <std_srvs/Empty.h>
	#include <laser_geometry/laser_geometry.h>
	#include "message_filters/subscriber.h"
	#include <tf/message_filter.h>




float frame_count=0;
sensor_msgs::PointCloud2 wall;
using namespace std ;
class LaserScanToPointCloud{

public:

  ros::NodeHandle n_;
  laser_geometry::LaserProjection projector_;
  tf::TransformListener listener_;
  message_filters::Subscriber<sensor_msgs::LaserScan> laser_sub_;
  tf::MessageFilter<sensor_msgs::LaserScan> laser_notifier_;
  ros::Publisher scan_pub_;
  //ros::Publisher scan_pub;

  LaserScanToPointCloud(ros::NodeHandle n) :
    n_(n),
    laser_sub_(n_, "/scan", 10),
    laser_notifier_(laser_sub_,listener_, "laser", 10)
  {

    laser_notifier_.registerCallback(
      boost::bind(&LaserScanToPointCloud::scanCallback, this, _1));

    laser_notifier_.setTolerance(ros::Duration(0.01));

    scan_pub_= n_.advertise<sensor_msgs::PointCloud2>("/input_cloud",1);


  }

  void scanCallback (const sensor_msgs::LaserScan::ConstPtr& scan_in)
  {
    sensor_msgs::PointCloud2 cloud;

    try
    {
    	//projector_.transformLaserScanToPointCloud(
    	//          "laser",*scan_in, cloud,listener_);
	projector_.projectLaser(*scan_in,cloud);

    }
    catch (tf::TransformException& e)
    {
        std::cout << e.what();
        return;
    }
    
    // Do something with cloud.

    for(unsigned int i=0; i<cloud.fields.size(); i++)
		{cloud.fields[i].count = 1;}
 

     if (::frame_count==0)
    {
		wall=cloud;	
		scan_pub_.publish(wall); //save 1st pointcloud as wall sensor_msg
		ROS_INFO("published wall");
    }
    else{

    	scan_pub_.publish(cloud);
		//ROS_INFO("published frame %f ",::frame_count);
    }
     ::frame_count++;
  }

};
     
    namespace cloud_assembler
    {

    typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

    class CloudAssembler
    {
     
    public:
      CloudAssembler();
      void cloudCallback(const sensor_msgs::PointCloud2& cloud);
     
    private:
      ros::NodeHandle node_;
     
      ros::ServiceServer pause_srv_;
     
      ros::Publisher output_pub_;
      ros::Subscriber cloud_sub_;
     
      tf::TransformListener tf_;
     
      PointCloudXYZ assembled_PointCloudXYZ;
      int buffer_length_;
      std::vector<sensor_msgs::PointCloud2> cloud_buffer_;
      bool assemblerPaused_;
     
      void addToBuffer(sensor_msgs::PointCloud2 cloud);
      void assembleCloud();
      bool pauseSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);
     
    };
     
    CloudAssembler::CloudAssembler()
    {
      ros::NodeHandle private_nh("~");
     
      private_nh.param("buffer_length", buffer_length_, 40);
     
      output_pub_ = node_.advertise<sensor_msgs::PointCloud2> ("/output_cloud", 100);
     
      pause_srv_ = node_.advertiseService("/pause_assembler", &CloudAssembler::pauseSrv, this);
     
      cloud_sub_ = node_.subscribe("/input_cloud", 100, &CloudAssembler::cloudCallback, this);
     
      PointCloudXYZ clear;
      assembled_PointCloudXYZ = clear;
     
      assemblerPaused_ = false;
    }
     
    void CloudAssembler::cloudCallback(const sensor_msgs::PointCloud2& cloud)
    {
      addToBuffer(cloud);
      assembleCloud();
     
      sensor_msgs::PointCloud2 cloud_msg;

      toROSMsg(assembled_PointCloudXYZ, cloud_msg);

      cloud_msg.header.frame_id = cloud.header.frame_id;
      cloud_msg.header.stamp = ros::Time::now();
     
     }
     
    void CloudAssembler::assembleCloud()
    {
      ROS_DEBUG("Assembling.");
     
      unsigned int i;
     
      if (assemblerPaused_)
      {
        ROS_INFO("assemblerPaused_ is true");
      }
      if (!assemblerPaused_)
      {
        ROS_DEBUG("assemblerPaused_ is false");
      }
     
      std::string fixed_frame = cloud_buffer_[0].header.frame_id;
      //ROS_INFO("call assembleCloud() for frame %f when cloud buffer has size %d",::frame_count-1, cloud_buffer_.size());

      PointCloudXYZ new_cloud;
      new_cloud.header.frame_id = fixed_frame;
      new_cloud.header.stamp = ros::Time::now();


      for (i = 0; i < cloud_buffer_.size(); i++)
      {
    	  PointCloudXYZ temp_cloud;

        fromROSMsg(cloud_buffer_[i], temp_cloud);
        temp_cloud.header.frame_id = fixed_frame;

        new_cloud += temp_cloud;

      }
     
      // If it's paused, don't overwrite the stored cloud with a new one, just keep publishing the same cloud
      if (!assemblerPaused_)
      {
        assembled_PointCloudXYZ = new_cloud;
      }
      else if (assemblerPaused_)
      {
        ROS_DEBUG("The Assembler will continue to publish the same cloud.");
      }
     
    }
     
    bool CloudAssembler::pauseSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp)
    {
      ROS_INFO("In service call: %s", assemblerPaused_?"true":"false");
     
      if (!assemblerPaused_)
      {
        ROS_INFO("Now paused.");
        assemblerPaused_ = true;
      }
      else if (assemblerPaused_)
      {
        assemblerPaused_ = false;
        ROS_DEBUG("Unpaused.");
      }
     
      return true;
    }
     
    void CloudAssembler::addToBuffer(sensor_msgs::PointCloud2 cloud)
    {
      //ROS_INFO("Adding cloud %f to buffer. Current buffer length is %d",::frame_count-1, cloud_buffer_.size());
      if (cloud_buffer_.size() >= (unsigned int)buffer_length_)
      {
        sensor_msgs::PointCloud2 cloud_msg;

        toROSMsg(assembled_PointCloudXYZ, cloud_msg);
    	output_pub_.publish(cloud_msg);
    	ROS_INFO("publish assembled cloud at frame %f ",::frame_count-1);
        cloud_buffer_.erase(cloud_buffer_.begin(),cloud_buffer_.end());

      }
      PointCloudXYZ temp_cloud;

              fromROSMsg(cloud, temp_cloud);


              for (int j=0 ; j<temp_cloud.points.size();j++){
            	  temp_cloud.points[j].z=::frame_count/18;
              }

              toROSMsg(temp_cloud, cloud);


      cloud_buffer_.push_back(cloud);
    }
     
     
    }; // namespace
     

int main(int argc, char** argv)
{

	  ros::init(argc, argv, "my_cloud_assembler");
          cout << "Poin 1";
	  ros::NodeHandle n;
          cout << "Poin 2";
	  LaserScanToPointCloud lstopc(n);
          cout << "Poin 3";
	  cloud_assembler::CloudAssembler cloud_assembler;
          cout << "Poin 3";
	  ros::spin();

	  return 0;
}
