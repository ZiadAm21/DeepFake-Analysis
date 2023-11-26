# DeepFakeDetective

Note: detector.py currently only runs on gpu.
To run:
1. clone this repo
2. create 2 files, DeepFrames and RawFrames, on the same level as detector.py
3. rewrite the rootdir in detector.py to your destination
4. if you would like to the detector on another video simple replace the one
5. the final output will be a list of confidence scores in the range [0, 1] where 0 is deepfake and 1 is real
6. the list will saved into a scores.txt file
7. the individual frame scores could be combined to give a final confidance score for the entire video
8. finally, the paper this project is based on boasts an AUC score of 99.9% on the Celeb-DF v2 dataset

This method uses heart rate estimation to detect deepfakes as detailed by the paper 'DeepFakesON-Phys: DeepFakes Detection based on Heart Rate Estimation'. This working princinple of this method is that deepfakes do not mimic the subtle change in skin tone cause by a person's heartbeat.

This project was made as a submissions to the JunctionX Budapest 2023 hackathon.

