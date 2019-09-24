import extract_features
import camera_calibration
import RANSAC


def main():
    print("Question 1:")
    print("The question 2 and 3 will be showed after close image window")
    print("-----------------------------")
    extract_features.main()
    print("Question 2:")
    print("-----------------------------")
    camera_calibration.main()
    print("Question 3:")
    print("-----------------------------")
    print("noise_version1")
    RANSAC.main("data/noise_version1.txt")
    print("-----------------------------")
    print("noise_version2")
    RANSAC.main("data/noise_version2.txt")


if __name__ == '__main__':
    main()