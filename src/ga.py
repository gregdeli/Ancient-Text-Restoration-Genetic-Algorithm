import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pygad

"""----Text Extraction and Vectorization-----"""

# Open dataset
df = pd.read_csv("Dataset/iphi2802.csv", delimiter="\t")

# Create a dataframe with the "text" column where "region_main_id" is 1693 (Greater Syria  and  the  East)
df = df[df["region_main_id"] == 1693]
texts_df = df[["text"]]

# Encode the text using tf-idf vectorization.
# Use the whole vocabulary to represent each text.
vectorizer = TfidfVectorizer()
vectorized_texts = vectorizer.fit_transform(texts_df["text"])

# Vocabulary with the tokenized terms
text_vocab = vectorizer.get_feature_names_out()

# Vectorize the damaged inscription
damaged_inscription = "αλεξανδρε ουδις"
damaged_inscription_vectorized = vectorizer.transform([damaged_inscription])

# Calculate the cosine similarity between the damaged inscription and the vectorized texts
cosine_similarities = cosine_similarity(
    damaged_inscription_vectorized, vectorized_texts
)

# Get the top-10 most similar texts
top_10_similar_texts = cosine_similarities.argsort()[0][::-1][:10]

# Print the top-10 most similar texts
print("\nTop 10 most similar texts:")
for i in top_10_similar_texts:
    print(f"\nText {i}: {texts_df.iloc[i]['text']}")


"""------------Genetic Algorithm------------"""


def add_sol_to_inscription(solution, damaged_inscription):
    # Convert the solution to integers
    solution = [int(i) for i in solution]

    # The output is the fixed inscription, using the two words given by the solution
    word1 = text_vocab[solution[0]]
    word2 = text_vocab[solution[1]]

    # Add the first word to the beginning of the damaged inscription and the second word to the end
    fixed_inscription = word1 + " " + damaged_inscription + " " + word2

    return fixed_inscription


def fitness(ga_instance, solution, solution_idx):
    # Add the solution to the damaged inscription
    fixed_inscription = add_sol_to_inscription(solution, damaged_inscription)

    # Vectorize the fixed inscription
    fixed_inscription_vectorized = vectorizer.transform([fixed_inscription])

    # Calculate the cosine similarity between the fixed inscription and the top 5 similar texts
    fitness_score = 0
    for i in top_10_similar_texts:
        fixed_inscription_similarity = cosine_similarity(
            fixed_inscription_vectorized, vectorized_texts[i]
        )
        # Add 1 to the fixed inscription similarity score in order to avoid negative values
        fixed_inscription_similarity += 1

        fitness_score += fixed_inscription_similarity[0][0]

    return fitness_score


# Parameters
num_generations = 500
population_size = 200
num_genes = 2
num_parents_mating = 5
mutation_probability = 0.01
crossover_probability = 0.6

# Define the gene space: integers between 1 and 1678
high = len(text_vocab) - 1
gene_space = {"low": 0, "high": high, "step": 1}

# Specify the selection technique
parent_selection_type = "rws"  # Use "rws", "rank" or "tournament"

# Specify the crossover technique
crossover_type = "single_point"  # Use "single_point", "two_points" or "uniform"

# Create an array of size num_generations to store the best fitness score of each generation from each execution
best_fitness_per_generation = [0] * num_generations
fitnesses_idx = 0


# Callback function called after each generation
def on_generation(ga_instance):
    global fitnesses_idx

    best_solution, best_solution_fitness, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )

    # Add the best fitness score of the current generation to the currently completed generations index in the array
    best_fitness_per_generation[fitnesses_idx] += best_solution_fitness
    fitnesses_idx += 1
    if fitnesses_idx >= num_generations:
        fitnesses_idx = 0


# PyGAD instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness,
    sol_per_pop=population_size,
    num_genes=num_genes,
    gene_space=gene_space,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    crossover_probability=crossover_probability,
    mutation_probability=mutation_probability,
    keep_elitism=2,
    on_generation=on_generation,
)

# Run the genetic algorithm 10 times and keep track of the fitness score of the best solution
average_fitness = 0
best_solution_fitness = 0
best_solution = None

for i in range(10):
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness > best_solution_fitness:
        best_solution = solution
    average_fitness += solution_fitness

# Print the best solution from all the runs
fixed_inscription = add_sol_to_inscription(best_solution, damaged_inscription)
print(f'\nFixed inscription: "{fixed_inscription}"')

# Print the average fitness score of the best solution
average_fitness /= 10
print(f"\nAverage fitness score of the best solution: {average_fitness}")

# Plot the fitness score of the best solution of each generation from all the runs

# Get the average fitness score of the best solution of each generation of each run
average_fitnesses_per_generation = np.array(best_fitness_per_generation) / 10

# Plot the average fitness score of the best solution of each generation
plt.plot(range(num_generations), average_fitnesses_per_generation)
plt.xlabel("Generation")
plt.ylabel("Average Best Fitness Score")
plt.title("Average Best Fitness Score Per Generation Across Runs")
plt.grid(True)
plt.show()
