---
title: Density-Based Spatial Clustering of Applications with Noise
date: 2022-09-12
categories:
  - Basic ML
tags: 
  - DBSCAN
  - Clustering
  - ML Analysis
  - Python
  - English
---

### 1. What is Density-Based Spatial Clustering of Applications with Noise
- DBSCAN (Density-based spatial clustering of applications with noise) uses density-based clustering among clustering algorithms.
- In the case of K-Means or Hierarchical clustering, clustering is performed using the distance between clusters. Density-based clustering is a method of clustering high-density parts because the dots are densely clustered.
- To put it simply, if there are ‘n’(or more) points within a radius ‘x’ of a certain point, it is recognized as a single cluster.
- For example, suppose that there is a point ‘p’, and if there are ‘m’(minPts) points within a distance ‘e’(epsilon) from the point ‘p’, it is recognized as a cluster.
- In this condition, that is, a point ‘p’ having ‘m’ points within a distance ‘e’ is called a core point.
- To use the DBSCAN algorithm, the distance epsilon value from the reference point and the number of points(minPts) within this radius, should be passed as a factor.
- In the figure below, if minPts = 4, if there are more than four points in the radius of epsilon around the blue point ‘P’, it can be judged as one cluster, and in the figure below ‘P’ becomes a core point because there are five points.
        
    ![](images/DBSCAN/Untitled.png)
        
- In the figure below, since the gray point P2 has three points within the epsilon radius based on the point P2, it does not reach minPts=4, so it does not become the core point of the cluster, but it is called a border point because it belongs to the cluster with the previous point P as the core point.
        
    ![](images/DBSCAN/Untitled%201.png)
        
- In the figure below, P3 becomes a core point because it has four points within the epsilon radius.
        
    ![](images/DBSCAN/Untitled%202.png)
        
- However, another core point P is included in the radius around P3, and in this case, core point P and P3 are considered to be connected and are grouped into one cluster.
        
    ![](images/DBSCAN/image1.png)
        
- Finally, P4 in the figure below is not included in the range that satisfies minPts=4 no matter what point is centered. In other words, it becomes an outlier that does not belong to any cluster, which is called noise point.

    ![](images/DBSCAN/image2.png)
        
- Putting it all together, we get the following picture:
        
    ![](images/DBSCAN/image3.png)
        
- In summary, if there are more points than the number of minPts in the epsilon radius around a point, it becomes a cluster around that point, and that point is called a core point.
- When a core point becomes part of a cluster of different core points, it became one big cluster.
- A point belonging to a cluster but not a core point by itself is called a border point, and is mainly a point forming the outer edge of the cluster. A point that does not belong to any cluster becomes a noise point.


### 2. Key Points
- The advantage of the DBSCAN algorithm is that it does not have to set the number of clusters like K Means, and because clusters are connected to each other according to the density of clusters, clusters with geometric shapes can be found well, and outlier detection is possible through noise point.
- It is oftenly used for learning but not in the field → If it is small data, you can use it, but because there is a lot of data in the field, use efficiency is low.
    
    ![](images/DBSCAN/image4.png)
    
- [Example of clustering geometry](https://en.wikipedia.org/wiki/DBSCAN)

- [Sample Code](https://github.com/bwcho75/dataanalyticsandML/blob/master/Clustering/5.%20DBSCANClustering-IRIS%204%20feature-Copy1.ipynb)




---
- [Korean reference](https://bcho.tistory.com/1205)