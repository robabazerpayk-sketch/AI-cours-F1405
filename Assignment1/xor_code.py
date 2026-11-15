from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# إنشاء شبكة عصبية بطبقة مخفية واحدة فيها نورونين
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='lbfgs', max_iter=1000, random_state=42)

# تدريب الشبكة
mlp.fit(X, y)

# توقع النتائج
predictions = mlp.predict(X)

print("Predictions:", list(predictions))
print("Final Weights (Hidden Layer):", mlp.coefs_[0])
print("Final Weights (Output Layer):", mlp.coefs_[1])
