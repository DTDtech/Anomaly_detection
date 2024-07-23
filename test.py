import numpy as np
from load_data import train_loader  
from vgg16_model import VGG16
from sklearn import svm
from load_data import NUM_EPOCHS
import os

from sklearn.decomposition import PCA
import matplotlib.font_manager
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# X = 0.3 * np.random.randn(5, 2)
# X_train = np.r_[X + 2, X - 2]
# # print(X_train)

# X_outliers = np.random.uniform(low=-4, high=4, size=(5, 2))
# print(X_outliers)

model = VGG16()

feature_list = []

# for class_name in os.listdir(root):
#     class_path = os.path.join(root, class_name) 
#     if os.path.isdir(class_path):
#         for image_name in os.listdir(class_path):
#             img_path = os.path.join(class_path, image_name)
#             image_paths.append(img_path)
#         for x in range(10):
#             print(image_paths[x])

for epoch in range(1):
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.view(3, images.size(0), 224, 224)
        if batch_id < 10:
            feature = model(images)
            feature = feature.detach().numpy()
            nsamples, depth, nx, ny = feature.shape
            feature = feature.reshape((nsamples, depth * nx * ny))
            feature_list.append(feature)


        # print(id)
        # print(label[id])
        # print(image[id])
        # print(image[id].size())


X_outliers = np.random.uniform(low=-4, high=4, size=(2, 50176))
# for feature in feature_list:
#     svm_model.fit(feature)
feature_list = np.vstack(feature_list)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_list)

svm_model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
svm_model.fit(reduced_features)

# Generate random outliers for testing
X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
y_pred_outliers = svm_model.predict(X_outliers)

# Plotting the decision boundary
_, ax = plt.subplots()
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
X = np.c_[xx.ravel(), yy.ravel()]

DecisionBoundaryDisplay.from_estimator(
    svm_model,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    levels=[0, 10000],
    colors="palevioletred",
)
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    X,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    levels=[0],
    colors="darkred",
    linewidths=2,
)

s = 40
b1 = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c="white", s=s, edgecolors="k")
c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")
plt.legend(
    [mlines.Line2D([], [], color="darkred"), b1, c],
    [
        "learned frontier",
        "training observations",
        "new abnormal observations",
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.show()



# print(feature_list)
# print(feature_list.size())

# output = model(x)
# print(output.size())