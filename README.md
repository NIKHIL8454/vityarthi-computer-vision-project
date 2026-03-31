Computer Vision is a field of Artificial Intelligence that enables machines to interpret and analyze visual data such as images and videos. In this project, computer vision techniques are applied to identify agricultural pests from images using deep learning.

The core of the system is based on Convolutional Neural Networks (CNNs), which are widely used for image classification tasks. CNNs automatically learn spatial features such as edges, textures, and shapes from images through multiple layers of convolution, pooling, and activation functions. This allows the model to distinguish between different pest categories based on visual patterns.

To improve performance and reduce training time, transfer learning is used. A pre-trained model (such as MobileNetV2) trained on large image datasets is adapted for pest classification. The earlier layers of the model capture general image features, while the later layers are fine-tuned to classify specific pest types.

The system performs the following steps:

The input image is resized and normalized for consistency.
The processed image is passed through the trained CNN model.
The model outputs a probability distribution across all pest classes using a softmax function.
The class with the highest probability is selected as the predicted pest.

A confidence threshold is applied to handle uncertain predictions. If the maximum probability is below a predefined threshold, the system classifies the input as “Not a pest.” This helps improve reliability when the input image does not belong to any trained category.

In addition to classification, the system integrates domain knowledge by associating each pest with textual information. Based on the predicted class, the system retrieves:

Symptoms or damage caused by the pest
Suggested remedies or control measures

This combination of deep learning and knowledge-based output transforms the system from a simple classifier into a practical decision-support tool for agricultural applications.
