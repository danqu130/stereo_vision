

 1. ./read

 2. ./calibrate -w 18 -h 13 -n 74 -s 0.025 -d "../img/1/" -i "left" -o "cam_left.yml" -e "jpg"

 3. ./calibrate -w 18 -h 13 -n 74 -s 0.025 -d "../img/2/" -i "right" -o "cam_right.yml" -e "jpg"

 4. ./calibrate_stereo -n 74 -u cam_left.yml -v cam_right.yml -L "../img/1/" -R "../img/2/" -l left -r right -o cam_stereo.yml

 5. ./undistort_rectify -l ../img/1/left1.jpg -r ../img/2/right1.jpg -c cam_stereo.yml -L left.jpg -R right.jpg