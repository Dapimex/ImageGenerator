from PIL import Image, ImageDraw, ImageColor
import random
import numpy


#   TO BE SET BY USER

image_name = "Plane.tiff"  # name of the file with the input image
#   NOTE: file should be in the same folder with the script, otherwise write the path
folder_name = "Plane"  # name of the folder to save the generated images in
#   NOTE: the folder should be created in the same folder with the script before execution
run_name = "Plane picture"    # the title of current execution, will be printed before the start
#   it helps to navigate between different parallel executions to check their states
background_color = "white"  # specify the name of the background color or web-code for it

#   MUTATION PROBABILITIES

mutation_probability_replicas = 1   # stands for probability of individual in population
# to be chosen for mutation (only for children after the crossover)
mutation_probability_dots = 0.00125  # stands for probability of dot in individual
# to be chosen for mutation (mutation is changing the color for random)

#   IMAGE CONFIGURATIONS

SIZE = 512  # size of one side of image in pixels
dot_size = 11   # diameter of dot in pixels

#   PROGRAM CONFIGURATIONS

size_of_population = 100  # amount of individuals in the population
generations = 7000  # number of generations when the program is finishing
save_ratio = 20  # number of iterations to save the intermediate result

initialImage = Image.open(image_name)  # type: Image.Image
# uploading the image to the program as the object of Image class
colors = initialImage.getcolors(SIZE*SIZE)  # set of colors, used in the initial image

# amount of pixels to offset from the side of image to be centralized
tab = (SIZE - (SIZE // dot_size * dot_size)) // 2   # in this case - exactly the middle of the population
crossover_border = size_of_population // 2  # the border between best and worst parts of population


# given the probability randomizer gives boolean value:
#   True: with probability_of_success
#   False: with probability 1 - probability_of_success
def custom_randomizer(probability_of_success):
    return random.random() < probability_of_success


# class Dot represents the circle, which the image consist of
# as a parameters there are the position of current dot and its color
class Dot:
    def __init__(self, x, y, color):
        self.color = color
        self.x = x
        self.y = y

    # choose the random color from the list of colors from the initial image
    # used for the mutation
    def set_random_color(self):
        self.color = random.choice(colors)[1]


# class Individual represents the image as the 2-dimensional array of dots with calculated fitness score
class Individual:
    def __init__(self):
        self.dots = []
        self.fitness_score = 100

    # filling the individual with random dots
    def generate_random_individual(self):
        for row_ind, row in enumerate(range(SIZE // dot_size)):
            row = []
            for col_ind, dot in enumerate(range(SIZE // dot_size)):
                row.append(Dot(col_ind, row_ind, random.choice(colors)[1]))
            self.dots.append(row)
        return self

    # given 2 individuals, process the crossover and come up with the child
    # indiv is a list of 2 individuals
    def cross_individual(self, indiv):
        for row_ind, row in enumerate(range(SIZE // dot_size)):
            row = []
            for col_ind, dot in enumerate(range(SIZE // dot_size)):
                row.append(Dot(col_ind, row_ind, indiv[random.randint(0, 1)].dots[row_ind][col_ind].color))
            self.dots.append(row)
        return self

    # convert the array of dots to the object of Image
    def create_image(self):
        generated_image = Image.new("RGB", [SIZE, SIZE], ImageColor.getrgb(background_color))
        d = ImageDraw.Draw(generated_image)
        for row_ind, row in enumerate(self.dots):
            for col_ind, dot in enumerate(row):
                init_coord_x = tab + col_ind * dot_size
                init_coord_y = tab + row_ind * dot_size
                fin_coord_x = init_coord_x + dot_size
                fin_coord_y = init_coord_y + dot_size
                d.ellipse([(init_coord_x, init_coord_y), (fin_coord_x, fin_coord_y)], dot.color)
        return generated_image

    # iterating through the dots and change the colors (according to probability)
    def mutate(self):
        for row_ind, row in enumerate(self.dots):
            for col_ind, dot in enumerate(row):
                if custom_randomizer(mutation_probability_dots):
                    dot.set_random_color()
        self.calculate_fitness_score()

    def calculate_fitness_score(self):
        global initialImage
        created_image = self.create_image()
        array_generated = numpy.array(created_image).astype(numpy.int)
        array_initial = numpy.array(initialImage).astype(numpy.int)
        self.fitness_score = (numpy.abs(array_generated - array_initial).sum() / 255.0 * 100) / array_initial.size
        return self


# class Population represents a set of individuals
class Population:
    def __init__(self):
        self.individuals = []

    # fill the population with the random individuals
    def generate_first_population(self):
        for i in range(size_of_population):
            self.individuals.append(Individual().generate_random_individual().calculate_fitness_score())
        self.sort_individuals()

    # returns the fittest individual from the set of individuals, according to the fitness score
    def get_fittest(self) -> Individual:
        self.sort_individuals()
        return self.individuals[0]

    # sort the array of individuals by fitness score
    def sort_individuals(self):
        self.individuals.sort(key=lambda val: val.fitness_score)

    # given a sorted array of individuals, drop out the worst half of population and process
    # the crossover on the best part, appending children back to the population
    def crossover(self):
        new_gen = self.individuals[:crossover_border]
        for i in range(crossover_border - 1):
            new_individual = Individual().cross_individual([new_gen[i], new_gen[i+1]]).calculate_fitness_score()
            new_gen.append(new_individual)
        new_individual = Individual().cross_individual([new_gen[crossover_border], new_gen[0]])\
            .calculate_fitness_score()
        new_gen.append(new_individual)
        self.individuals = new_gen

    # given the array of individuals after the crossover, take the last half which is a new children and mutate them
    def mutate(self):
        for index, individual in enumerate(self.individuals[crossover_border:]):
            if custom_randomizer(mutation_probability_replicas):
                individual.mutate()
        self.sort_individuals()


#   STARTING THE PROGRAM

print(run_name)  # name the execution to understand then, what is it (useful for parallel execution)
# print the configurations for this run
print("Mutation Dots: " + str(mutation_probability_dots))
print("Mutation Individuals: " + str(mutation_probability_replicas))
print("Dot size: " + str(dot_size))
print("Population: " + str(size_of_population))
print("Generations: " + str(generations))

population = Population()   # initialize the initial population
population.generate_first_population()  # fill the population with random individuals
# then iterate crossover-mutation cycle for the preset amount of generations
for i in range(generations):
    print("This is iteration num: " + str(i))
    population.crossover()
    population.mutate()
    print(population.individuals[0].fitness_score)
    # save intermediate result after "save_ratio" amount of iterations
    if i % save_ratio == 0:
        individual_to_print = population.get_fittest()
        image_to_print = individual_to_print.create_image()
        # every newly created image is saved in the folder "folder_name"
        # and have the name "contest-%generation%_%fitness_score%.png"
        image_to_print.save(folder_name + "/contest-" + str(i) + "_" +
                            str(individual_to_print.fitness_score) + ".png")
