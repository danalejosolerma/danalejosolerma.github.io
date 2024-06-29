# Data scientist | Computational physicist
  
### PROFESSIONAL EXPERIENCE
 **TotalEnergies, Palaiseau, France**   
  	*Trainee - Solar Photovoltaics (May 2024 - Present)*

  - I am currently developing a machine learning model for the forecasting of solar irradiance in solar farms using data recollected from satellite images, sky imagers and irradiance historical data.

  _Tech stack: Python, Pytorch, TensorFlow, Scikit-learn, OpenCV_

  **Onepoint, Paris, France**   
	 *Consultant data engineering/AI (June 2022 - Jan 2023)* 
	 
   - Conducted an in-depth analysis and documentation of an internal database aimed at optimizing the recruitment process, resulting in an improvement in candidate selection.
   - Mapped and analyzed a comprehensive database to develop a robust framework for estimating market size and market share across the Americas for a multinational corporation.

  _Tech stack: Python, Excel_

  **LEEL lab, CEA, Saclay, France**   
	 *Research Engineer - Machine Learning (Jun 2020 - Oct 2021)*
  
   - Co-developed a ML python code for the prediction of chemical composition and thickness of mineral samples, obtaining accuracies between 2% and 10% depending on the quality of the simulated data.
   - Developed a ML python code for the analysis of hyperspectral images and implemented algorithms for anomaly detection, reducing the analysis time by a factor of 20.
   - Construction, training, testing, optimization and deployment of neural networks.
   - Presented at conferences and wrote scientific publications.  
  
   _Tech stack: Python, Bash, Keras, Scikit-learn, Scikit-image, MPI_

  **SPINTEC lab, CEA, Grenoble, France**   
	 *Physics researcher - Theory/Simulation of Spintronics (Dec 2016 - Nov 2020)*
	
   - Developed a mathematical model for spin transfer torque in a graphene based spintronic device.
   - Conducted simulations in python of spintronic devices using a high performance computing machine.
   - Performed analysis, interpretation and data visualisation of results using python.
   - Presented at conferences and wrote scientific publications

   _Tech stack: Python, Fortran, C, Bash, MPI, Scipy, matplotlib, seaborn_
   
  **UNICAMP, Campinas, Brazil**   
	  *Research assistant - Molecular dynamics (Aug 2015 - Jul 2016)*
	  
   - Derived an equation to simulate the spiral shape of a carbon nanoscroll.
   - Conducted molecular dynamics simulations of nanoscrolls using LAMMPS software on a high-performance computing machine.
   - Presented at conferences and wrote a scientific publication.

  _Tech stack: Bash, Matlab, LAMMPS_

### EDUCATION
    
  - Specialized master in Data science | ENSAE (May 2024)
  - PhD. in Physics | Grenoble Alpes University (May 2021)
  - MSc. in Physics | Campinas State University (Aug 2015)
  - BSc. in Physics | University of Valle (May 2013 )

## Projects
### Minimal Diffusion
[Repository](https://github.com/danalejosolerma/Segmentation_satellite_images)
In this project, it was trained a deep diffusion model (DDPM) by creating a simplified version of a U-Net architecture and using various datasets (MNIST and SVHN). The training process allowed the DDPM to learn how to generate images from pure noise by observing a wide range of examples.
<p align="center">
  <img src="https://github.com/danalejosolerma/portfolio/blob/main/assets/img/gif-mnist.gif?raw=true" alt="MNIST" width="274" height="274" />
  <img src="https://github.com/danalejosolerma/portfolio/blob/main/assets/img/gif-mnist.gif?raw=true" alt="SVHN" width="274" height="274" />
</p>

### Segmentation of Satellite Images
[Repository](https://github.com/danalejosolerma/Segmentation_satellite_images)
Convolutional neural network built using Keras developed to classify different objects from RGB aerial images The dataset used to train this convolutional neural network was obtained from Kaggle https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery that contains 6 different classes: Building, Land, Road, Vegetation, Water and Unlabeled.

The results obtained are a first good approximation of what is wanted for this task. During the training it was obtained a Jaccard similarity (SIMRATIO) of 0.7925, which is pretty decent for the simplicity of the model.

### Unsupervised Gender Classification Using Word Embeddings
[Repository](https://github.com/danalejosolerma/NLP-gender-project)
The primary objective of this report is to predict the gender based on personal data from an extract of data of the French census from the years 1836 to 1936. In this dataset we do not have access to the actual labels, so I approached this task as a unsupervised text classification and at end I compare the results obtained with the predictions provided. The gender inference problem is of vital importance in fields such as anthropological and sociological research and the treatment of personal data will be done respecting ethics principles and current legislation. In this work the subject of gender is considered as binary, in consonance with the age of this dataset, which by no means implies a political statement by the author.

Several methods exist to efficiently classify text, going from very simple ones like Naive Bayes, going to more advance methods like Deep learning models, Transformer models and Word embeddings. The latter was the chosen method in this project.

