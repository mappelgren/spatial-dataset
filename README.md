# Dataset Generation for Spatial Games

This code generates the datasets described in TBD. 

The purpose is to create datasets for signaling games played between two artificial agents. The signaling games has the first agent see two images, a target and distractor and then send a message to describe the target image. The listener also sees two images and must pick the correct one.

Thus, when generating the data we generate a series of pairs of scenes, target and distractor. 

In addition, the goal of the project is to allow for the agents to potentially see the scenes from different directions. To do this, we rotate the camera around and capture each scene from 4 different angles, 0, 90, 180, and 270. We do this for both scenes. This creates an output structure that looks like this:

    Dataset-name/
        Target/
            0/
                img1.png
                img2.png
            90/
                img1.png
                img2.png
            180/
                img1.png
                img2.png
            270/
                img1.png
                img2.png
        Distractor/
            0/
                img1.png
                img2.png
            90/
                img1.png
                img2.png
            180/
                img1.png
                img2.png
            270/
                img1.png
                img2.png
  
There are a few different functions which can generate the images. There is one function for creating simpler datasets with single objects where the variation between target and distractor are on visual attributes. 
The othe function is to create datasets which have the same objects but have mirrored spatial attributes. This is to test for the ability to describe spatial language. 

The code in this repository is based on the scene generation code of the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/) as described in the paper:
**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
<br>
<a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
<a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
<a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
<a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
<a href='http://larryzitnick.org/'>Larry Zitnick</a>,
<a href='http://www.rossgirshick.info/'>Ross Girshick</a>
<br>
Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)


