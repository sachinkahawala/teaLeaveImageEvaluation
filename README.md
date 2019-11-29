# teaLeaveImageEvaluation
 Evaluating a image of tea plantation that is taken from a drone.
 
 Following methodology is proposed to hilight the tea buds from other leaves.
-> ![Image of Yaktocat](https://user-images.githubusercontent.com/25722196/69854725-cf04da80-12af-11ea-8b1b-8d27862a0bbd.png) <-
Following input image is taken through the above process and the results are being showed and explained in this section.

![image](https://user-images.githubusercontent.com/25722196/69854815-0a070e00-12b0-11ea-92ae-891dea871d97.png)

For the input image first step was to equalize the histogram to enhance the features of the image.

![image](https://user-images.githubusercontent.com/25722196/69854836-19865700-12b0-11ea-8d41-5e58ac8ce8ee.png)

As we can see from the histogram equalized image tea buds are further highlighted. Furthermore, the edge details have being enhanced. One slight disadvantage here is that the effect of sunlight also has being enhanced. 
Next step is to apply the HLS colour filter on the histogram equalized RGB image.

![image](https://user-images.githubusercontent.com/25722196/69854913-53575d80-12b0-11ea-8cd9-cffb40aef857.png)

As we can see here tea buds are further highlighted with a bright yellow colour. Since this is a three-colour space if we try to get a clusteration on this it can be prone to errors. For an example, since we are using Euclidean distance to get the distance between two points, ({0,0,255},{0,255,0},{255,0,0}) all these pixels have same Euclidean distance from pixel ({0,0,0} or {255,255,255}). To avoid this possible error the option was to go for a single channel image for clusteration. This further reduces the computing power as well. Following images shows the HLS channels separately. 

![image](https://user-images.githubusercontent.com/25722196/69855044-9dd8da00-12b0-11ea-900d-beed69444a08.png)

If we look carefully on the separate H, L and S channels we can see that S channel (Saturation) has the bud details close to the HLS converted image, but it has some random noise. So, the target was to cluster the saturation channel image and then compensate the noise using above mentioned techniques. 

![image](https://user-images.githubusercontent.com/25722196/69855075-b648f480-12b0-11ea-9b00-0596351a6de5.png)

Above image is the results that were received when clustering Saturation space image with k =3. A modified K-Means algorithm were designed to successfully segment tea buds from tea leaves. K-means algorithm were modified in such a way where we can initialize the starting centroids.  After many experiments and explorations, it was found that when K=3 and [0,127,255] centroid selected as initial centroids, the buds of the separated tea leaves could be well identified.
Following figure is the cluster results when applied on the original image. 

![image](https://user-images.githubusercontent.com/25722196/69855130-ceb90f00-12b0-11ea-9bf8-c7824056eddf.png)

As we can see from the above image tea buds are clearly captured but there are lot of noise in the clustered image.  Next step is to reduce the noise with mentioned noise filtration techniques. 
Final results is shown below.

![image](https://user-images.githubusercontent.com/25722196/69855195-f7d99f80-12b0-11ea-88d9-c8f461081546.png)

Following table shows a comparison between the Ground truth and the results recieved.

![image](https://user-images.githubusercontent.com/25722196/69855291-3707f080-12b1-11ea-8783-c587416879ac.png)

