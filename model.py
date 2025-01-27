import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go


# Load normalized dataset from NCBI GEO database and prepare it for training
data_path = "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE252145&format=file&file=GSE252145_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"
data = pd.read_csv(data_path, compression="gzip", sep="\t")
data = data.iloc[: , 1:]
data = data.T
data.fillna(0)

# Generate labels, pre treatment = 0, post treatment = 1
# Labels are generated based on the sample names, and each pre-treatment sample is even, and each post-treatment sample is odd
labels = data.index.tolist()
for label in labels:
    if int(label[3:]) % 2 == 0:
        labels[labels.index(label)] = 0
    else:
        labels[labels.index(label)] = 1
#print(labels)

label_names = {
    0: "Pre-treatment",
    1: "Post-treatment"
}

# Utilize the PCA algorithm to reduce the dimensionality of the data
pca = PCA(n_components=31)
reduced_data = pca.fit_transform(data)
#print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")

# Split the data into training and testing sets
data_train = reduced_data[:int(len(reduced_data) * 0.8)]
labels_train = labels[:int(len(labels) * 0.8)]
data_test = reduced_data[int(len(reduced_data) * 0.8):]
labels_test = labels[int(len(labels) * 0.8):]

# Convert the data and labels to PyTorch tensors
data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.int64)
data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64)

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(data_train_tensor, labels_train_tensor)
test_dataset = torch.utils.data.TensorDataset(data_test_tensor, labels_test_tensor)

# Create PyTorch data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple neural network model with three fully connected layers
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(31, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Train the model
model = Model()
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
losses = []
for epoch in range(num_epochs):
    
    model.train()
    for data, labels in train_loader:
        outputs = model(data.to(device))
        loss = criterion(outputs, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():

    predictions = []
    actuals = []
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        predictions += predicted.tolist()
        actuals += batch_y.tolist()
    
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    cm = confusion_matrix(actuals, predictions)

    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(cm)

torch.save(model.state_dict(), 'GSE252145.pth')

# Create a confusion matrix visualization
cm_df = pd.DataFrame(cm, index=["Actual Pre-Treatment", "Actual Post-Treatment"], columns=["Predicted Pre-Treatment", "Predicted Post-Treatment"])
cm_fig = px.imshow(
    cm_df,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=cm_df.columns,
    y=cm_df.index,
    text_auto=True,
    color_continuous_scale="Greens",
)

cm_fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
)

# Create a training loss plot
epochs = list(range(1, num_epochs + 1))
loss_df = pd.DataFrame(losses, columns=["Loss"])
loss_plot_fig = go.Figure()
loss_plot_fig.add_trace(
    go.Scatter(
        x=epochs,
        y=loss_df["Loss"],
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="blue"),
    )
)
loss_plot_fig.update_layout(
    title="Training Loss Over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Loss",
)

app = dash.Dash(__name__)

# Create the Dash app layout
app.layout = html.Div(children=[

    html.H1("GSE252145 RNA-Seq Pre/Post-Treatment Classification"),

    html.Div(children=[
        html.H1("Training Loss Plot"),
        dcc.Graph(figure=loss_plot_fig),
    ]),

    html.Div(children=[
        html.H1("Confusion Matrix Visualization"),
        dcc.Graph(figure=cm_fig),
        html.H2(f"Accuracy: {accuracy * 100:.2f}%"),
        html.H2(f"Precision: {precision:.2f}"),
        html.H2(f"Recall: {recall:.2f}"),
        html.H2(f"F1 Score: {f1:.2f}"),
    ])]
)

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)

