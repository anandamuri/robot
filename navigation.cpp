#include <cmath>
#include <iostream>
#include <signal.h>

#include <mbot_bridge/robot.h>

#include <mbot_lib/behaviors.h>
#include <mbot_lib/controllers.h>
#include <mbot_lib/utils.h>


bool ctrl_c_pressed;
void ctrlc(int) {
    ctrl_c_pressed = true;
}


int main() {
    signal(SIGINT, ctrlc);
    signal(SIGTERM, ctrlc);

    // Initialize the robot.
    mbot_bridge::MBot robot;
    // Reset the robot odometry to zero.
    robot.resetOdometry();
    // *** Task: Implement bug navigation finite state machine *** / /
    float x, y, theta;

    std::cout << "Enter your goal x-coordinate" << std::endl;
    std::cin >> x;
    std::cout << "Enter your goal y-coordinate" << std::endl;
    std::cin >> y;
    std::cout << "Enter your goal theta heading" << std::endl;
    std::cin >> theta;

    std::vector<float> goal = {x, y, theta};

    while(!ctrl_c_pressed) {
        
        std::vector<float> ranges;
        std::vector<float> thetas;
        robot.readLidarScan(ranges, thetas);

        std::vector<float> currentPose = robot.readOdometry();
        bool withinPoseTolerance = (std::abs(currentPose[0] - goal[0]) < 0.01) && (std::abs(currentPose[1] - goal[1]) < 0.01);
        bool withinOrientationTolerance = (std::abs(normalizeAngle(currentPose[2] - goal[2])) < 0.15);

        if (withinPoseTolerance && withinOrientationTolerance) {
            break;
        }

        if (isGoalAngleObstructed(goal, robot.readOdometry(), ranges, thetas)) {
            std::vector<float> avoid = computeWallFollowerCommand(ranges, thetas);
            robot.drive(avoid[0], avoid[1], avoid[2]);
            std::cout << "Wall Following" << std::endl;
        } else {
            std::vector<float> goalCoords = computeDriveToPoseCommand(goal, robot.readOdometry());
            robot.drive(goalCoords[0], goalCoords[1], goalCoords[2]);
            std::cout << "Driving to goal" << std::endl;
        }

        std::cout << "Current x-pose = " << currentPose[0] << std::endl;
        std::cout << "Current y-pose = " << currentPose[1] << std::endl;
        std::cout << "Current theta-pose = " << currentPose[2] << std::endl;

        if (ctrl_c_pressed) {
            break;
        }
    }

    // Stop the robot.
    robot.stop();

    std::vector<float> finalPose = robot.readOdometry();
    std::cout << "Final MBot odometry x-pose = " << finalPose[0] 
              << ", y = " << finalPose[1] 
              << ", theta = " << finalPose[2] 
              << std::endl;

    bool withinPoseTolerance = (std::abs(finalPose[0] - goal[0]) < 0.01) && (std::abs(finalPose[1] - goal[1]) < 0.01);
    bool withinOrientationTolerance = (std::abs(normalizeAngle(finalPose[2] - goal[2])) < 0.15);
    //Checking if final odometry position passes tolerance criteria
    if ((withinPoseTolerance) && (withinOrientationTolerance)) {
        std::cout << "Your final pose is within 1 centimeter and 0.15 radians of your goal!" << std::endl;
        std::cout << "Final pose x = " << finalPose[0]
                  << "Final pose y = " << finalPose[1] 
                  << "Final pose theta = " << finalPose[2] << std::endl;
    } else {
        std::cout << "Your final pose is not within either the pose or orientation tolerance!" << std::endl;
        std::cout << "Here is your error difference, x = " << finalPose[0] - goal[0]
                  << ", y = " << finalPose[1] - goal[1]             
                  << ", theta = " << finalPose[2] - goal[2] << std::endl;
    }
    return 0;
}
