/**
 * Controls the robot to maintain a given distance to the wall directly in front of it.
 * This code uses Bang-Bang control or P-control to maintain the setpoint distance.
 */

#include <cmath>
#include <iostream>
#include <signal.h>
#include <mbot_bridge/robot.h>
#include <mbot_lib/controllers.h>
#include <mbot_lib/utils.h>

bool ctrl_c_pressed;
void ctrlc(int)
{
    ctrl_c_pressed = true;
}

int main(int argc, const char *argv[])
{
    signal(SIGINT, ctrlc);
    signal(SIGTERM, ctrlc);

    mbot_bridge::MBot robot;
    
    //Vectors storing different distances and angles the LiDar scanner returns.
    std::vector<float> ranges;
    std::vector<float> thetas;

    //Setpoint is how far the robot is from the user/object.
    float setpoint = 0.4; 

    //Amount of error that can happen.
    float tolerance = 0.1;

    //Directional control.
    float scaling = -0.5;
    
    //Proportional constant.
    float kp = 0.5;

    while (true) {
        //Reading values indefinetly.
        robot.readLidarScan(ranges, thetas);

        //Uses function findFwdDist to find distance to the wall.
        float dist_to_wall = findFwdDist(ranges, thetas);

        //Condition for robot to not get too close.
        if (dist_to_wall < 0) continue;

        //Printing distance to wall.
        std::cout << "Distance to wall: " << dist_to_wall << std::endl;

        //Using the bangBangControl to control robot speed and direction relative to where it is.
        float controlSignal = bangBangControl(dist_to_wall, setpoint, scaling, tolerance);
        std::cout << "Control Signal: " << controlSignal << std::endl;  // Debugging output
        robot.drive(controlSignal, 0.0, 0.0);

        //Stops robot execution.
        if (ctrl_c_pressed)
            break;
    }

    // Stop the robot.
    robot.stop();
    return 0;
}
