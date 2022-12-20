import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit import pyplot

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)


# Set the seed for reproducibility
np.random.seed(42)

# Define a function to plot the distribution and the sampling distribution
def plot_distributions(distribution, sample_size, num_samples):
    # Generate the data
    data = distribution(size=num_samples)
    
    # Plot the distribution
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Distribution')
    
    # Initialize an array to store the means of the samples
    means = np.zeros(num_samples)
    
    # Generate the samples and calculate the mean for each sample
    for i in range(num_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        means[i] = np.mean(sample)
    
    # Plot the sampling distribution
    plt.hist(means, bins=50, density=True, alpha=0.5, label='Sampling distribution')
    
    # Add a legend and show the plot
    plt.legend()
    plt.show()
    st.pyplot()

# Define the main function
def main():
    # Set the title
    st.title('Central Limit Theorem Demonstration')
    st.write("by: A. Vera")
    
    # Select the distribution
    distribution = st.selectbox('Select the distribution:', ['Uniform', 'Normal', 'Exponential', 'Poisson'])
    
    # Select the sample size
    sample_size = st.slider('Sample size:', min_value=1, max_value=1000, value=50, step=1)
    
    # Select the number of samples
    num_samples = st.slider('Number of samples:', min_value=1, max_value=1000, value=500, step=1)
    
    # Plot the distributions
    if st.button('Plot distributions'):
        if distribution == 'Uniform':
            plot_distributions(np.random.uniform, sample_size, num_samples)
        elif distribution == 'Normal':
            plot_distributions(np.random.normal, sample_size, num_samples)
        elif distribution == 'Exponential':
            plot_distributions(np.random.exponential, sample_size, num_samples)
        elif distribution == 'Poisson':
            plot_distributions(np.random.poisson, sample_size, num_samples)

# Run the app
if __name__ == '__main__':
    main()
