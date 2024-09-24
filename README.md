

#### Support Vector Machine (SVM) for Neutron Interaction Classification

This project applies a Support Vector Machine (SVM) model to classify neutron interaction events using data from the Modular Neutron Array (MoNA) experiment at the Facility for Rare Isotope Beams (FRIB).

#### **Project Overview**
The goal of this project is to investigate applications of machine learning to nuclear physics, specifically in classifying neutron trajectories. The SVM model was chosen for its ability to handle high-dimensional feature spaces and complex boundaries in neutron interaction data. The work directly contributes to the Modular Neutron Array (MoNA) Collaboration, supporting ongoing research in neutron-unbound systems.

#### **Files Included**
- **svm.ipynb**: Jupyter notebook implementing the SVM model, including data preprocessing, training, validation, and evaluation.
- **data/**: Directory containing the neutron interaction dataset used in training the SVM model ().


#### **Requirements**
To run this project, the following dependencies are required:
- Python 3.x
- Jupyter Notebook
- NumPy
- Scikit-learn
- Matplotlib

You can install the dependencies using:
```bash
pip install numpy scikit-learn matplotlib
```

#### **How to Run the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/Rmot1202/svm-neutron-classification.git
   ```
2. Navigate to the project folder:
   ```bash
   cd svm-neutron-classification
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook svm.ipynb
   ```
4. Run the notebook cells to train, validate, and evaluate the SVM model.

#### **Project Highlights**
- **Data Preprocessing**: The dataset was normalized and preprocessed to optimize the performance of the SVM classifier.
- **Model Training**: The SVM was trained using a radial basis function (RBF) kernel to maximize separation between neutron interaction events.
- **Evaluation**: Cross-validation was employed to ensure the robustness of the model, and evaluation was performed using metrics like accuracy, precision, recall, and F1-score.

#### **Future Improvements**
- Experiment with different SVM kernel functions.
- Enhance the dataset for broader generalization.
- Integrate the SVM model into the MoNA analysis software pipeline.

#### **Contributors**
- Raven Mott – Developer
- Dr. Thomas Redpath – Project Supervisor (Virginia State University Nuclear Science Lab)

This work is part of a larger project that explores the use of machine learning methods in nuclear physics experiments, contributing to the Modular Neutron Array (MoNA) and the Large multi-Institutional Scintillating Array (LISA) collaboration at FRIB. It provides hands-on experience in machine learning, data analysis, and scientific communication.

