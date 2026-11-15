X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1] 

def activate(value):
    return 1 if value >= 0 else 0

def train_perceptron(w1, w2, b, learning_rate=0.1, epochs=10):
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            linear_output = (w1 * X[i][0]) + (w2 * X[i][1]) + b
            prediction = activate(linear_output)
            error = y[i] - prediction
            
            if error != 0:
                w1 = w1 + (learning_rate * error * X[i][0])
                w2 = w2 + (learning_rate * error * X[i][1])
                b = b + (learning_rate * error)
                errors += 1
                
        print(f"Epoch {epoch+1}: w1={w1}, w2={w2}, b={b}, errors={errors}")
        
        if errors == 0:
            print(f"Final parameters: w1={w1}, w2={w2}, b={b}")
            break

train_perceptron(8, 0, 0)
train_perceptron(8, 0, 4)
train_perceptron(8, -2, 4)
