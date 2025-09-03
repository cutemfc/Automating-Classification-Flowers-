# 🎄Automating-Classification-Flowers-
# 🎄Project: Information Technology. Automating Flower Classification for a Start-up Using Deep Learning
# 🎄Goal: The goal is to build a machine learning model capable of classifying images of flowers into their respective species.
# 🎄Dataset:
(1) 102 flower categories with diverse species.

(2) 40 to 258 images per category, leading to class imbalance challenges.

(3) Training, Validation and test splits for model evaluation.

(4) Images with varying sizes, lighting conditions, and backgrounds, making classification more challenging.

note: redefine the training, and test, we use test_datest as training_set because the number is 6000 better than ~1000

#🎄Model Aracitecture:
Base model: ResNet50 (pretrained on ImageNet, top layers excluded)

• Dual-output heads:

• Coarse classification: 10 clusters

• Fine classification: 102 classes

• Hierarchical setup allows shared feature learning while optimizing both coarse and fine predictions

# 🎄Performance Comparison:
We held two methods to improve the model
(1) Apply Gaussain blur and flipes of the pictures
| Method                                      | Training Coarse Acc. | Training Fine Acc. | Validation Coarse Acc. | Validation Fine Acc. |
| ------------------------------------------- | -------------------- | ------------------ | ---------------------- | -------------------- |
| **Gaussian blur + Hierarchical clustering** | 0.4344               | 0.1412             | 0.2176                 | 0.0294               |
| **Resize only + Hierarchical clustering**   | 0.5258               | 0.3506             | 0.3333                 | 0.1196               |
| **Original model (no clustering)**          | –                    | 0.0085             | –                      | 0.0098               |
| **U²-Net + mean/std preprocessing**         | –                    | 0.1576             | –                      | 0.0873               |

(1) Gaussianblur + Hierachical clustering and Hierachical clustering only

Hierarchical clustering provides ~10× improvement in fine accuracy compared to the original model.

Resizing only + clustering achieves the best overall results.

Gaussian blur degrades performance, likely due to the loss of fine-grained floral features needed for classification.

(2) Using Pytorch and OpenCv to do segmentation and turn to Tensorflow (apply U2-Net)

U²-Net with mean/std normalization improves fine-grained classification performance compared to the original model, despite not using clustering.
Training fine accuracy reaches 0.1576, and validation fine accuracy improves to 0.0873, indicating that statistical normalization helps the model better generalize to unseen data.
While not outperforming clustering-based methods, this preprocessing strategy enhances feature consistency, likely aiding segmentation and classification of subtle floral patterns.
The results suggest that U²-Net’s segmentation capabilities combined with normalization offer a meaningful boost, especially when clustering is not applied.
# 🎄Skill uses
• Data preprocessing (OpenCV,Pytorch, Numpy,Pandas,Scikit-learn )
Image resizing, flipping, Gaussian blur, segmentation (U²-Net), normalization, clustering, dataset restructuring.
• 	Model building (TensorFlow,Keras,PyTorch )
Transfer learning with ResNet50, dual-output heads for coarse/fine classification, hierarchical architecture.
• 	Evaluation (Scikit-learn,Numpy,Matplotlib)
Accuracy tracking, validation strategy, performance comparison, visualization of results.
