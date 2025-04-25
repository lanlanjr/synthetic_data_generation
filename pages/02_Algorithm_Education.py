import streamlit as st
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="Algorithm Education",
        page_icon="🤖",
        layout="wide"
    )

def page_introduction():
    """Display the introduction section of the page"""
    st.title("Machine Learning Algorithm Education 🎓")
    
    st.markdown("""
    Welcome to the Algorithm Education page! This interactive guide helps you understand various machine learning 
    algorithms used in classification tasks. Each algorithm is explained in detail with:

    - 📝 Clear descriptions and explanations
    - ✅ Advantages and limitations
    - 🎯 Practical use cases
    - 📊 Mathematical foundations
    - 💻 Implementation examples
    - 🔬 Interactive demonstrations
    - 📚 Academic references
    
    ### How to Use This Guide
    1. Select an algorithm from the dropdown menu below
    2. Explore its characteristics and implementation details
    3. Try the interactive demo with different datasets
    4. Compare performance metrics and visualizations
    
    ### Available Algorithms
    This guide covers popular classification algorithms including:
    - Naive Bayes variants
    - Support Vector Machines
    - Neural Networks
    - Tree-based methods
    - Nearest Neighbors
    - Linear Classifiers
    - Ensemble Methods
    
    ### Why Understanding Algorithms Matters
    Choosing the right algorithm for your machine learning task is crucial for:
    - Achieving optimal performance
    - Efficient resource utilization
    - Meeting specific problem constraints
    - Understanding model behavior and limitations
    """)

def algorithm_info():
    """Display detailed algorithm information"""
    # First show the introduction
    page_introduction()
    
    algorithms = {
        "Gaussian Naive Bayes (GaussianNB)": {
            "description": """
            A probabilistic classifier based on Bayes' theorem with strong independence assumptions between features. 
            Assumes features follow a Gaussian (normal) distribution.
            """,
            "pros": [
                "Simple and fast",
                "Works well with small datasets",
                "Good for high-dimensional data",
                "Performs well when features are normally distributed"
            ],
            "cons": [
                "Assumes feature independence (often unrealistic)",
                "Limited by Gaussian distribution assumption",
                "May underperform when features are highly correlated"
            ],
            "use_cases": [
                "Text classification",
                "Spam detection",
                "Medical diagnosis",
                "Real-time prediction scenarios"
            ],
            "math_details": {
                "main_formula": r"""
                P(y|x_1,...,x_n) = \frac{P(y)\prod_{i=1}^{n}P(x_i|y)}{P(x_1,...,x_n)}
                """,
                "component_formulas": [
                    {
                        "name": "Gaussian Probability Density",
                        "formula": r"""
                        P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma^2_y}\right)
                        """
                    },
                    {
                        "name": "Class Prior Probability",
                        "formula": r"""
                        P(y) = \frac{\text{number of samples in class y}}{\text{total number of samples}}
                        """
                    }
                ],
                "explanation": """
                - P(y|x₁,...,xₙ) is the posterior probability of class y given features
                - P(y) is the prior probability of class y
                - P(xᵢ|y) is the likelihood of feature xᵢ given class y
                - μy and σ²y are the mean and variance of features in class y
                """
            },
            "references": [
                {
                    "title": "Naive Bayes and Text Classification",
                    "authors": "Sebastian Raschka",
                    "publication": "arXiv preprint",
                    "year": "2014",
                    "url": "https://arxiv.org/abs/1410.5329"
                },
                {
                    "title": "scikit-learn: Machine Learning in Python",
                    "authors": "Pedregosa et al.",
                    "publication": "Journal of Machine Learning Research",
                    "year": "2011",
                    "url": "https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html"
                },
                {
                    "title": "Fundamental Mathematical Formulas Used in Machine Learning",
                    "authors": "Showmik Setta",
                    "publication": "Medium",
                    "year": "2023",
                    "url": "https://medium.com/@showmiklovesport/fundamental-mathematical-formulas-used-in-machine-learning-beginner-21c0843e61e0"
                }
            ]
        },
        "Linear Support Vector Classification (LinearSVC)": {
            "description": """
            A linear classifier that finds the hyperplane that best separates classes by maximizing the margin between them.
            Optimized implementation of Support Vector Classification for linear classification.
            """,
            "pros": [
                "Effective for high-dimensional spaces",
                "Memory efficient",
                "Faster than standard SVC with linear kernel",
                "Works well when classes are linearly separable"
            ],
            "cons": [
                "Only suitable for linear classification",
                "Sensitive to feature scaling",
                "May struggle with overlapping classes",
                "No probability estimates by default"
            ],
            "use_cases": [
                "Text classification",
                "Image classification",
                "Bioinformatics",
                "High-dimensional data analysis"
            ],
            "math_details": {
                "main_formula": r"""
                \min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \max(0, 1-y_i(w^Tx_i+b))
                """,
                "component_formulas": [
                    {
                        "name": "Decision Function",
                        "formula": r"""
                        f(x) = w^Tx + b
                        """
                    },
                    {
                        "name": "Margin Width",
                        "formula": r"""
                        \text{margin} = \frac{2}{||w||}
                        """
                    }
                ],
                "explanation": """
                - w is the weight vector
                - b is the bias term
                - C is the regularization parameter
                - yᵢ are the true labels (±1)
                - xᵢ are the input features
                """
            },
            "references": [
                {
                    "title": "A Tutorial on Support Vector Machines for Pattern Recognition",
                    "authors": "Christopher J.C. Burges",
                    "publication": "Data Mining and Knowledge Discovery",
                    "year": "1998",
                    "url": "https://link.springer.com/article/10.1023/A:1009715923555"
                },
                {
                    "title": "Support Vector Machines",
                    "authors": "Andrew Ng",
                    "publication": "CS229 Lecture Notes, Stanford University",
                    "year": "2018",
                    "url": "http://cs229.stanford.edu/notes/cs229-notes3.pdf"
                },
                {
                    "title": "Machine Learning Algorithms: Mathematical Deep Dive",
                    "authors": "Vidushi Meel",
                    "publication": "viso.ai",
                    "year": "2021",
                    "url": "https://viso.ai/deep-learning/machine-learning-algorithms-mathematical-guide/"
                }
            ]
        },
        "Support Vector Classification (SVC)": {
            "description": """
            A powerful classifier that can perform non-linear classification using different kernel functions to transform 
            the feature space. Creates an optimal hyperplane in a transformed feature space.
            """,
            "pros": [
                "Effective for non-linear classification",
                "Works well with high-dimensional data",
                "Robust against overfitting",
                "Versatile through different kernel functions"
            ],
            "cons": [
                "Computationally intensive for large datasets",
                "Sensitive to feature scaling",
                "Kernel selection can be challenging",
                "Memory intensive for large datasets"
            ],
            "use_cases": [
                "Image classification",
                "Handwriting recognition",
                "Bioinformatics",
                "Pattern recognition"
            ],
            "math_details": {
                "main_formula": r"""
                \min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \xi_i
                """,
                "component_formulas": [
                    {
                        "name": "Kernel Function (RBF)",
                        "formula": r"""
                        K(x,x') = \exp\left(-\gamma ||x-x'||^2\right)
                        """
                    },
                    {
                        "name": "Decision Function",
                        "formula": r"""
                        f(x) = \sum_{i=1}^{n} \alpha_i y_i K(x_i,x) + b
                        """
                    }
                ],
                "explanation": """
                - K(x,x') is the kernel function
                - γ is the kernel coefficient
                - αᵢ are the dual coefficients
                - ξᵢ are the slack variables
                """
            },
            "references": [
                {
                    "title": "Support Vector Networks",
                    "authors": "Cortes C., Vapnik V.",
                    "publication": "Machine Learning",
                    "year": "1995",
                    "url": "https://link.springer.com/article/10.1007/BF00994018"
                },
                {
                    "title": "A Practical Guide to Support Vector Classification",
                    "authors": "Hsu, Chang, and Lin",
                    "publication": "BJU International",
                    "year": "2003",
                    "url": "https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf"
                },
                {
                    "title": "Machine Learning Algorithms: Mathematical Deep Dive",
                    "authors": "Vidushi Meel",
                    "publication": "viso.ai",
                    "year": "2021",
                    "url": "https://viso.ai/deep-learning/machine-learning-algorithms-mathematical-guide/"
                }
            ]
        },
        "Multi-layer Perceptron (MLPClassifier)": {
            "description": """
            A neural network classifier that learns non-linear models by training multiple layers of nodes. 
            Each node uses a non-linear activation function to transform inputs.
            """,
            "pros": [
                "Can learn highly non-linear patterns",
                "Capable of learning complex relationships",
                "Good generalization with proper regularization",
                "Can handle multiple classes naturally"
            ],
            "cons": [
                "Requires careful hyperparameter tuning",
                "Computationally intensive",
                "Sensitive to feature scaling",
                "May get stuck in local minima"
            ],
            "use_cases": [
                "Image recognition",
                "Speech recognition",
                "Complex pattern recognition",
                "Financial prediction"
            ],
            "math_details": {
                "main_formula": r"""
                h_l = \sigma(W_l h_{l-1} + b_l)
                """,
                "component_formulas": [
                    {
                        "name": "ReLU Activation",
                        "formula": r"""
                        \sigma(x) = \max(0,x)
                        """
                    },
                    {
                        "name": "Softmax Output",
                        "formula": r"""
                        P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
                        """
                    }
                ],
                "explanation": """
                - hₗ is the output of layer l
                - Wₗ is the weight matrix for layer l
                - bₗ is the bias vector for layer l
                - σ is the activation function
                """
            },
            "references": [
                {
                    "title": "Learning representations by back-propagating errors",
                    "authors": "Rumelhart, D. E., Hinton, G. E., & Williams, R. J.",
                    "publication": "Nature",
                    "year": "1986",
                    "url": "https://www.nature.com/articles/323533a0"
                },
                {
                    "title": "Gradient-based learning applied to document recognition",
                    "authors": "LeCun Y., Bottou L., Bengio Y., & Haffner P.",
                    "publication": "Proceedings of the IEEE",
                    "year": "1998",
                    "url": "https://ieeexplore.ieee.org/document/726791"
                },
                {
                    "title": "Fundamental Mathematical Formulas Used in Machine Learning",
                    "authors": "Showmik Setta",
                    "publication": "Medium",
                    "year": "2023",
                    "url": "https://medium.com/@showmiklovesport/fundamental-mathematical-formulas-used-in-machine-learning-beginner-21c0843e61e0"
                }
            ]
        },
        "Extra Trees Classifier": {
            "description": """
            An ensemble method that builds multiple randomized decision trees and averages their predictions. 
            Similar to Random Forest but with additional randomization in the tree-building process.
            """,
            "pros": [
                "Lower variance than Random Forest",
                "Faster training than Random Forest",
                "Good at handling high-dimensional data",
                "Less prone to overfitting"
            ],
            "cons": [
                "May have slightly lower accuracy than Random Forest",
                "Can be memory intensive",
                "Less interpretable than single decision trees",
                "May require more trees than Random Forest"
            ],
            "use_cases": [
                "Feature selection",
                "Large dataset classification",
                "Remote sensing",
                "Biomedical classification"
            ],
            "math_details": {
                "main_formula": r"""
                \hat{f}_{et}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x)
                """,
                "component_formulas": [
                    {
                        "name": "Random Split Selection",
                        "formula": r"""
                        \text{gain}(s,D) = \frac{|D_l|}{|D|}H(D_l) + \frac{|D_r|}{|D|}H(D_r)
                        """
                    },
                    {
                        "name": "Entropy",
                        "formula": r"""
                        H(D) = -\sum_{k=1}^K p_k\log(p_k)
                        """
                    }
                ],
                "explanation": """
                - B is the number of trees
                - fᵦ is the prediction of the b-th tree
                - Dₗ and Dᵣ are left and right splits
                - pₖ is the proportion of class k in the node
                """
            },
            "references": [
                {
                    "title": "Extremely randomized trees",
                    "authors": "Geurts P., Ernst D., & Wehenkel L.",
                    "publication": "Machine Learning",
                    "year": "2006",
                    "url": "https://link.springer.com/article/10.1007/s10994-006-6226-1"
                },
                {
                    "title": "scikit-learn: Machine Learning in Python",
                    "authors": "Pedregosa et al.",
                    "publication": "Journal of Machine Learning Research",
                    "year": "2011",
                    "url": "https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html"
                },
                {
                    "title": "Fundamental Mathematical Formulas Used in Machine Learning",
                    "authors": "Showmik Setta",
                    "publication": "Medium",
                    "year": "2023",
                    "url": "https://medium.com/@showmiklovesport/fundamental-mathematical-formulas-used-in-machine-learning-beginner-21c0843e61e0"
                }
            ]
        },
        "Random Forest Classifier": {
            "description": """
            An ensemble learning method that constructs multiple decision trees and combines their predictions. 
            Each tree is built using a random subset of features and bootstrap samples of the data.
            """,
            "pros": [
                "Robust against overfitting",
                "Handles non-linear relationships well",
                "Provides feature importance",
                "Works well with high-dimensional data"
            ],
            "cons": [
                "Can be computationally intensive",
                "Less interpretable than single decision trees",
                "Memory intensive for large datasets",
                "May overfit on noisy datasets"
            ],
            "use_cases": [
                "Credit risk assessment",
                "Medical diagnosis",
                "Market prediction",
                "Image classification"
            ],
            "math_details": {
                "main_formula": r"""
                \hat{f}_{rf}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x)
                """,
                "component_formulas": [
                    {
                        "name": "Random Split Selection",
                        "formula": r"""
                        \text{gain}(s,D) = \frac{|D_l|}{|D|}H(D_l) + \frac{|D_r|}{|D|}H(D_r)
                        """
                    },
                    {
                        "name": "Entropy",
                        "formula": r"""
                        H(D) = -\sum_{k=1}^K p_k\log(p_k)
                        """
                    }
                ],
                "explanation": """
                - B is the number of trees
                - fᵦ is the prediction of the b-th tree
                - Dₗ and Dᵣ are left and right splits
                - pₖ is the proportion of class k in the node
                """
            },
            "references": [
                {
                    "title": "Random Forests",
                    "authors": "Breiman L.",
                    "publication": "Machine Learning",
                    "year": "2001",
                    "url": "https://link.springer.com/article/10.1023/A:1010933404324"
                },
                {
                    "title": "An Introduction to Statistical Learning",
                    "authors": "James G., Witten D., Hastie T., & Tibshirani R.",
                    "publication": "Springer",
                    "year": "2013",
                    "url": "https://www.statlearning.com/"
                },
                {
                    "title": "Machine Learning Algorithms: Mathematical Deep Dive",
                    "authors": "Vidushi Meel",
                    "publication": "viso.ai",
                    "year": "2021",
                    "url": "https://viso.ai/deep-learning/machine-learning-algorithms-mathematical-guide/"
                }
            ]
        },
        "K-Nearest Neighbors (KNeighborsClassifier)": {
            "description": """
            A non-parametric method that classifies a data point based on the majority class of its k nearest neighbors 
            in the feature space. Simple but effective algorithm.
            """,
            "pros": [
                "Simple to understand and implement",
                "No training phase",
                "Naturally handles multi-class cases",
                "Non-parametric (no assumptions about data)"
            ],
            "cons": [
                "Computationally intensive for large datasets",
                "Sensitive to irrelevant features",
                "Requires feature scaling",
                "Memory intensive (stores all training data)"
            ],
            "use_cases": [
                "Recommendation systems",
                "Pattern recognition",
                "Data imputation",
                "Anomaly detection"
            ],
            "math_details": {
                "main_formula": r"""
                \hat{f}_{knn}(x) = \frac{1}{k}\sum_{i=1}^k y_i
                """,
                "component_formulas": [
                    {
                        "name": "Distance Function",
                        "formula": r"""
                        d(x,x') = \sum_{i=1}^p |x_i - x'_i|^2
                        """
                    },
                    {
                        "name": "Decision Function",
                        "formula": r"""
                        f(x) = \text{sign}\left(\sum_{i=1}^k y_i \cdot \text{weight}(d(x,x_i))\right)
                        """
                    }
                ],
                "explanation": """
                - d(x,x') is the distance function
                - xᵢ are the k nearest neighbors
                - yᵢ are the labels of the k nearest neighbors
                - weight(d(x,x')) is the weight function based on distance
                """
            },
            "references": [
                {
                    "title": "Nearest Neighbor Pattern Classification",
                    "authors": "Cover T. & Hart P.",
                    "publication": "IEEE Transactions on Information Theory",
                    "year": "1967",
                    "url": "https://ieeexplore.ieee.org/document/1053964"
                },
                {
                    "title": "A Survey of Nearest Neighbor Techniques",
                    "authors": "Bhatia N. & Vandana",
                    "publication": "International Journal of Computer Science and Information Security",
                    "year": "2010",
                    "url": "https://arxiv.org/abs/1007.0085"
                },
                {
                    "title": "Machine Learning Algorithms: Mathematical Deep Dive",
                    "authors": "Vidushi Meel",
                    "publication": "viso.ai",
                    "year": "2021",
                    "url": "https://viso.ai/deep-learning/machine-learning-algorithms-mathematical-guide/"
                }
            ]
        },
        "Ridge Classifier": {
            "description": """
            A linear classifier that uses L2 regularization to prevent overfitting. Similar to logistic regression 
            but with different loss function and regularization.
            """,
            "pros": [
                "Good for multicollinear data",
                "Less prone to overfitting",
                "Computationally efficient",
                "Works well with many features"
            ],
            "cons": [
                "Only for linear classification",
                "May underfit complex patterns",
                "Sensitive to feature scaling",
                "No probability estimates"
            ],
            "use_cases": [
                "High-dimensional data classification",
                "Text classification",
                "Gene expression analysis",
                "Simple binary classification"
            ],
            "math_details": {
                "main_formula": r"""
                \min_{w} ||Xw - y||^2_2 + \alpha ||w||^2_2
                """,
                "component_formulas": [
                    {
                        "name": "Decision Function",
                        "formula": r"""
                        f(x) = w^Tx
                        """
                    },
                    {
                        "name": "L2 Penalty",
                        "formula": r"""
                        \text{penalty} = \alpha ||w||^2_2 = \alpha \sum_{j=1}^p w_j^2
                        """
                    }
                ],
                "explanation": """
                - w is the weight vector
                - α is the regularization strength
                - X is the feature matrix
                - y is the target vector
                - p is the number of features
                """
            },
            "references": [
                {
                    "title": "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
                    "authors": "Hoerl A.E. & Kennard R.W.",
                    "publication": "Technometrics",
                    "year": "1970",
                    "url": "https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634"
                },
                {
                    "title": "The Elements of Statistical Learning",
                    "authors": "Hastie T., Tibshirani R., & Friedman J.",
                    "publication": "Springer",
                    "year": "2009",
                    "url": "https://web.stanford.edu/~hastie/ElemStatLearn/"
                },
                {
                    "title": "Fundamental Mathematical Formulas Used in Machine Learning",
                    "authors": "Showmik Setta",
                    "publication": "Medium",
                    "year": "2023",
                    "url": "https://medium.com/@showmiklovesport/fundamental-mathematical-formulas-used-in-machine-learning-beginner-21c0843e61e0"
                }
            ]
        },
        "Multinomial Naive Bayes": {
            "description": """
            A specialized version of Naive Bayes for multinomially distributed data. Commonly used for text 
            classification with word counts.
            """,
            "pros": [
                "Fast training and prediction",
                "Works well with high-dimensional data",
                "Good for text classification",
                "Handles multiple classes well"
            ],
            "cons": [
                "Assumes feature independence",
                "Requires non-negative features",
                "Sensitive to feature distribution",
                "May underperform with continuous data"
            ],
            "use_cases": [
                "Document classification",
                "Spam detection",
                "Language detection",
                "Topic modeling"
            ],
            "math_details": {
                "main_formula": r"""
                P(y|x) = \frac{P(y)\prod_{i=1}^n P(x_i|y)}{\sum_{k} P(y_k)\prod_{i=1}^n P(x_i|y_k)}
                """,
                "component_formulas": [
                    {
                        "name": "Feature Probability",
                        "formula": r"""
                        P(x_i|y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}
                        """
                    },
                    {
                        "name": "Log Probability",
                        "formula": r"""
                        \log P(y|x) = \log P(y) + \sum_{i=1}^n \log P(x_i|y)
                        """
                    }
                ],
                "explanation": """
                - Nyᵢ is the count of feature i in class y
                - Ny is the total count of all features in class y
                - α is the smoothing parameter
                - n is the number of features
                """
            },
            "references": [
                {
                    "title": "A comparison of event models for naive Bayes text classification",
                    "authors": "McCallum A. & Nigam K.",
                    "publication": "AAAI-98 Workshop on Learning for Text Categorization",
                    "year": "1998",
                    "url": "https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf"
                },
                {
                    "title": "An empirical study of the naive Bayes classifier",
                    "authors": "Rish I.",
                    "publication": "IJCAI 2001 Workshop on Empirical Methods in Artificial Intelligence",
                    "year": "2001",
                    "url": "https://www.researchgate.net/publication/228845263_An_Empirical_Study_of_the_Naive_Bayes_Classifier"
                },
                {
                    "title": "Fundamental Mathematical Formulas Used in Machine Learning",
                    "authors": "Showmik Setta",
                    "publication": "Medium",
                    "year": "2023",
                    "url": "https://medium.com/@showmiklovesport/fundamental-mathematical-formulas-used-in-machine-learning-beginner-21c0843e61e0"
                }
            ]
        },
        "AdaBoost Classifier": {
            "description": """
            An ensemble method that builds a strong classifier by iteratively adding weak learners, focusing on 
            previously misclassified examples.
            """,
            "pros": [
                "Good generalization",
                "Less prone to overfitting",
                "Can identify hard-to-classify instances",
                "Works well with weak learners"
            ],
            "cons": [
                "Sensitive to noisy data and outliers",
                "Sequential nature (can't parallelize)",
                "Can be computationally intensive",
                "May require careful tuning"
            ],
            "use_cases": [
                "Face detection",
                "Object recognition",
                "Medical diagnosis",
                "Fraud detection"
            ],
            "math_details": {
                "main_formula": r"""
                F(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
                """,
                "component_formulas": [
                    {
                        "name": "Weak Learner Weight",
                        "formula": r"""
                        \alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
                        """
                    },
                    {
                        "name": "Sample Weight Update",
                        "formula": r"""
                        w_{i,t+1} = w_{i,t}\exp(-y_i\alpha_th_t(x_i))
                        """
                    }
                ],
                "explanation": """
                - hₜ(x) is the weak learner prediction
                - αₜ is the weight of weak learner t
                - εₜ is the weighted error rate
                - wᵢ,ₜ is the weight of sample i at iteration t
                """
            },
            "references": [
                {
                    "title": "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting",
                    "authors": "Freund Y. & Schapire R.E.",
                    "publication": "Journal of Computer and System Sciences",
                    "year": "1997",
                    "url": "https://www.sciencedirect.com/science/article/pii/S002200009791504X"
                },
                {
                    "title": "Experiments with a New Boosting Algorithm",
                    "authors": "Freund Y. & Schapire R.E.",
                    "publication": "International Conference on Machine Learning",
                    "year": "1996",
                    "url": "https://icml.cc/Conferences/1996/papers/boosting.pdf"
                },
                {
                    "title": "Machine Learning Algorithms: Mathematical Deep Dive",
                    "authors": "Vidushi Meel",
                    "publication": "viso.ai",
                    "year": "2021",
                    "url": "https://viso.ai/deep-learning/machine-learning-algorithms-mathematical-guide/"
                }
            ]
        }
    }

    # Add implementation details to each algorithm
    for algo_name in algorithms:
        algorithms[algo_name]["implementation"] = {
            "Gaussian Naive Bayes (GaussianNB)": {
                "code": """
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Initialize and train the model
gnb = GaussianNB()
gnb.fit(X, y)

# Make predictions
y_pred = gnb.predict(X)
                """,
                "key_parameters": {
                    "var_smoothing": "Portion of the largest variance of all features that is added to variances for calculation stability",
                    "priors": "Prior probabilities of the classes"
                },
                "tips": [
                    "Normalize features if they have very different scales",
                    "Good as a baseline model for comparison",
                    "Check feature distributions - should be roughly Gaussian"
                ]
            },
            "Linear Support Vector Classification (LinearSVC)": {
                "code": """
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
svc = LinearSVC(random_state=42, max_iter=1000)
svc.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "C": "Regularization parameter (default=1.0)",
                    "max_iter": "Maximum iterations for convergence",
                    "dual": "Dual or primal formulation"
                },
                "tips": [
                    "Always scale your features",
                    "Increase max_iter if model doesn't converge",
                    "Try different C values using cross-validation"
                ]
            },
            "Support Vector Classification (SVC)": {
                "code": """
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
svc = SVC(random_state=42)
svc.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "C": "Regularization parameter (default=1.0)",
                    "kernel": "Kernel function used to transform the data",
                    "gamma": "Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels"
                },
                "tips": [
                    "Always scale your features",
                    "Try different kernels and gamma values",
                    "Increase C if model underfits",
                    "Decrease C if model overfits"
                ]
            },
            "Multi-layer Perceptron (MLPClassifier)": {
                "code": """
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
mlp = MLPClassifier(random_state=42)
mlp.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "hidden_layer_sizes": "Number of neurons in each layer",
                    "activation": "Activation function used in the hidden layers",
                    "solver": "Optimization algorithm used to train the model",
                    "alpha": "L2 regularization parameter"
                },
                "tips": [
                    "Always scale your features",
                    "Try different activation functions",
                    "Increase hidden_layer_sizes if model underfits",
                    "Decrease hidden_layer_sizes if model overfits"
                ]
            },
            "Extra Trees Classifier": {
                "code": """
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
et = ExtraTreesClassifier(random_state=42)
et.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "n_estimators": "Number of trees in the forest",
                    "max_depth": "Maximum depth of the trees",
                    "min_samples_split": "Minimum number of samples required to split an internal node",
                    "min_samples_leaf": "Minimum number of samples required to be at a leaf node"
                },
                "tips": [
                    "Always scale your features",
                    "Try different max_depth values",
                    "Increase n_estimators if model underfits",
                    "Decrease n_estimators if model overfits"
                ]
            },
            "Random Forest Classifier": {
                "code": """
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "n_estimators": "Number of trees in the forest",
                    "max_depth": "Maximum depth of the trees",
                    "min_samples_split": "Minimum number of samples required to split an internal node",
                    "min_samples_leaf": "Minimum number of samples required to be at a leaf node"
                },
                "tips": [
                    "Always scale your features",
                    "Try different max_depth values",
                    "Increase n_estimators if model underfits",
                    "Decrease n_estimators if model overfits"
                ]
            },
            "K-Nearest Neighbors (KNeighborsClassifier)": {
                "code": """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
knn = KNeighborsClassifier()
knn.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "n_neighbors": "Number of neighbors to use",
                    "weights": "Weight function used in prediction",
                    "algorithm": "Algorithm used to compute the nearest neighbors",
                    "leaf_size": "Maximum number of samples in each leaf"
                },
                "tips": [
                    "Always scale your features",
                    "Try different n_neighbors values",
                    "Increase leaf_size if model underfits",
                    "Decrease leaf_size if model overfits"
                ]
            },
            "Ridge Classifier": {
                "code": """
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
ridge = RidgeClassifier(random_state=42)
ridge.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "alpha": "Regularization parameter (default=1.0)",
                    "solver": "Optimization algorithm used to train the model",
                    "max_iter": "Maximum number of iterations for the solver to converge"
                },
                "tips": [
                    "Always scale your features",
                    "Try different alpha values",
                    "Increase max_iter if model doesn't converge",
                    "Decrease max_iter if model overfits"
                ]
            },
            "Multinomial Naive Bayes": {
                "code": """
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
nb = MultinomialNB()
nb.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "alpha": "Regularization parameter (default=1.0)",
                    "fit_prior": "Whether to learn class prior probabilities or not",
                    "class_prior": "Prior probabilities of the classes"
                },
                "tips": [
                    "Always scale your features",
                    "Try different alpha values",
                    "Increase alpha if model underfits",
                    "Decrease alpha if model overfits"
                ]
            },
            "AdaBoost Classifier": {
                "code": """
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the model
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_scaled, y)
                """,
                "key_parameters": {
                    "n_estimators": "Number of trees in the forest",
                    "learning_rate": "Learning rate used to update the weights of the weak classifiers",
                    "algorithm": "Optimization algorithm used to train the model"
                },
                "tips": [
                    "Always scale your features",
                    "Try different learning_rate values",
                    "Increase n_estimators if model underfits",
                    "Decrease n_estimators if model overfits"
                ]
            }
        }.get(algo_name, {})

    # Algorithm selector
    selected_algo = st.selectbox(
        "Select an algorithm to learn more:",
        list(algorithms.keys())
    )

    # Display algorithm information
    if selected_algo:
        st.header(selected_algo)
        
        # Description
        st.subheader("Description")
        st.write(algorithms[selected_algo]["description"])
        
        # Two-column layout for pros and cons
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Advantages")
            for pro in algorithms[selected_algo]["pros"]:
                st.markdown(f"✅ {pro}")
                
        with col2:
            st.subheader("Disadvantages")
            for con in algorithms[selected_algo]["cons"]:
                st.markdown(f"⚠️ {con}")
        
        # Use cases
        st.subheader("Common Use Cases")
        for use_case in algorithms[selected_algo]["use_cases"]:
            st.markdown(f"🎯 {use_case}")

        # Add mathematical details section
        st.markdown("---")
        display_math_details(algorithms[selected_algo])

        # Add visual separator
        st.markdown("---")

        # Implementation section
        if "implementation" in algorithms[selected_algo]:
            st.subheader("Implementation Example")
            
            # Code example
            st.code(algorithms[selected_algo]["implementation"]["code"], language="python")
            
            # Key Parameters
            st.subheader("Key Parameters")
            for param, desc in algorithms[selected_algo]["implementation"]["key_parameters"].items():
                st.markdown(f"**`{param}`**: {desc}")
            
            # Implementation Tips
            st.subheader("Implementation Tips")
            for tip in algorithms[selected_algo]["implementation"]["tips"]:
                st.markdown(f"💡 {tip}")

        # Add interactive demo section
        st.subheader("Interactive Demo")
        if st.checkbox("Show Interactive Demo"):
            st.write("Select dataset:")
            dataset_choice = st.selectbox(
                "Choose a sample dataset",
                ["Iris", "Breast Cancer", "Wine", "Digits"]
            )

            if st.button("Run Demo"):
                try:
                    with st.spinner("Running demo..."):
                        demo_results = run_algorithm_demo(selected_algo, dataset_choice)
                        
                        # Display results
                        st.write("Model Performance:")
                        st.write(f"Accuracy: {demo_results['accuracy']:.4f}")
                        
                        # Show confusion matrix
                        st.write("Confusion Matrix:")
                        st.pyplot(demo_results['confusion_matrix_plot'])
                        
                        # Show learning curve
                        st.write("Learning Curve:")
                        st.pyplot(demo_results['learning_curve_plot'])
                except Exception as e:
                    st.error(f"Error running demo: {str(e)}")

        # Add a references section to display in the UI
        if st.checkbox("Show References"):
            st.subheader("Academic References")
            if "references" in algorithms[selected_algo]:
                for ref in algorithms[selected_algo]["references"]:
                    st.markdown(f"**{ref['title']}**")
                    st.markdown(f"*{ref['authors']}* ({ref['year']})")
                    st.markdown(f"Published in: {ref['publication']}")
                    st.markdown(f"[Link to Publication]({ref['url']})")
                    st.markdown("---")
            else:
                st.write("No references available for this algorithm.")

def run_algorithm_demo(algorithm_name, dataset_name):
    """Run a demo of the selected algorithm on the chosen dataset."""
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load dataset
    dataset_loaders = {
        "Iris": load_iris,
        "Breast Cancer": load_breast_cancer,
        "Wine": load_wine,
        "Digits": load_digits
    }
    
    data = dataset_loaders[dataset_name]()
    X, y = data.data, data.target
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = get_model_instance(algorithm_name)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions and accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_plot = plt.gcf()
    plt.close()
    
    # Create learning curve plot
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    lc_plot = plt.gcf()
    plt.close()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix_plot': cm_plot,
        'learning_curve_plot': lc_plot
    }

def get_model_instance(algorithm_name):
    """Return an instance of the specified algorithm."""
    models = {
        "Gaussian Naive Bayes (GaussianNB)": GaussianNB(),
        "Linear Support Vector Classification (LinearSVC)": LinearSVC(random_state=42),
        "Support Vector Classification (SVC)": SVC(random_state=42),
        "Multi-layer Perceptron (MLPClassifier)": MLPClassifier(random_state=42),
        "Extra Trees Classifier": ExtraTreesClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors (KNeighborsClassifier)": KNeighborsClassifier(),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42)
    }
    return models[algorithm_name]

def display_math_details(algorithm):
    """Display mathematical details for the algorithm."""
    if "math_details" in algorithm:
        st.subheader("Mathematical Details")
        
        # Main formula
        st.write("Main Formula:")
        st.latex(algorithm["math_details"]["main_formula"])
        
        # Component formulas
        st.write("Component Formulas:")
        for component in algorithm["math_details"]["component_formulas"]:
            st.write(f"**{component['name']}:**")
            st.latex(component["formula"])
        
        # Explanation
        st.write("**Variable Explanations:**")
        st.markdown(algorithm["math_details"]["explanation"])

if __name__ == "__main__":
    setup_page_config()
    algorithm_info()
