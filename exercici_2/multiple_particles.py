import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, Callable
import pyblock

class TrialWaveFunction:

    def __init__(self, alpha, lamb):
        self.alpha = alpha
        self.lamb = lamb

    def probability_ratio(self, X1, X2):
        raise NotImplementedError()
    

class FRepresentation(TrialWaveFunction):

    def __init__(self, alpha, lamb):
        super().__init__(alpha, lamb)

    def f1(self, x): return np.exp(-self.alpha*x*x)
        
    def f2(self, x): return np.abs(x)**self.lamb

    def wave_function(self, X):
        single_body = np.prod(self.f1(X))
        differences = X[:, None] - X[None, :]

        i, j = np.triu_indices(len(X), k=1)
        multiple_body = np.prod(self.f2(differences[i, j]))

    def pdf(self, X): return self.wave_function(self, X)**2

    def probability_ratio(self, X1, X2): return self.pdf(X1)/self.pdf(X2)


class URepresentation(TrialWaveFunction):

    def __init__(self, alpha, lamb):
        super().__init__(alpha, lamb)

    def u1(self, x): return -self.alpha*x*x

    def u2(self, x): return self.lamb*np.log(np.abs(x))

    

class MultipleParticlesInHarmonicTrap:

    # For this experiment, we'll consider that all the particles have the same mass
    def __init__(self, alpha, mass, temperature, omega, N_particles, lamb):
        self.mass = mass
        self.temperature = temperature
        self.omega = omega

        self.alpha = alpha
        self.N_particles = N_particles
        self.lamb = lamb

        self.H_BAR = 1.0

        # self.optimal_sigma, _ = self.find_optimal_sigma(verbose=False)



    def u1(self, x): return -self.alpha*x*x

    def u2(self, x): return self.lamb*np.log(np.abs(x))

    def wave_function(self, X: np.array): 
        single_body = np.prod(self.f1(X))
        differences = X[:, None] - X[None, :]
        multiple_body = np.prod()


    # Returns True or False depending on whether the sample was accepted or rejected.
    def sample_pdf(self, prev_sample, sigma: float):
        new_sample = prev_sample + random.gauss(mu=0, sigma=sigma)
        
        w = (self.wave_function(new_sample)**2)/(self.wave_function(prev_sample)**2)
        u = random.uniform(0,1)

        if u < min(1.0, w):
            return new_sample, True
        else:
            return prev_sample, False


    def optimal_sigma_plots(self, sigmas, probs):
        print(len(sigmas))
        print(len(probs))
        plt.figure()
        plt.plot(sigmas, 0.5-(np.array(probs)-0.5)**2)
        plt.xlabel("sigma")
        plt.ylabel("acceptance probability")
        plt.show()


    def find_optimal_sigma(self, verbose=False):
        N = 1000

        # Generate a list of 100 lineally spaced sigma values between 0 and 5
        sigmas = np.linspace(0, 5, 100).tolist()
        probs = []
    
        for sigma in sigmas: 
            # Samples is a list of N tuples (x, prob).
            
            sample = 0
            n_success = 0
            for i in range(N):
                sample, success = self.sample_pdf(sample, sigma)
                n_success += 1 if success else 0

            probs.append(n_success/N)

        optimal_index = np.argmin((np.array(probs)-0.5)**2)

        if verbose: self.optimal_sigma_plots(sigmas, probs)

        return sigmas[optimal_index], probs[optimal_index]


    def samples_generator(self, N) -> Generator[float]:
        sample = 0
        for i in range(N):
            sample, _ = self.sample_pdf(sample, sigma=self.optimal_sigma)
            yield sample


    def plot_histogram(self, histogram, min, max):
        bin_edges = np.linspace(min, max, len(histogram) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.bar(bin_centers, histogram, width=bin_edges[1] - bin_edges[0])
        plt.xlabel("Sample value")
        plt.ylabel("Frequency")
        plt.show()


    def draw_density_profile(self, nbins, _min, _max, N):
        histogram = [0 for i in range(nbins)]

        for sample in self.samples_generator(N):
            i = int(nbins * (sample - _min) / (_max - _min))
            i = min(i, nbins - 1)
            histogram[i] += 1

        self.plot_histogram(histogram, _min, _max)

        return histogram


    def potential_energy(self, x):
        return 0.5*self.mass*(self.omega*self.omega)*x*x


    def average_potential_energy(self, N):
        sum = 0
        for sample_x in self.samples_generator(N):
            sum += self.potential_energy(sample_x)
        return sum/N


    def local_kinetic_energy(self, x):
        return (2*self.alpha*x)**2 - 2*self.alpha

    
    def average_kinetic_energy_direct_estimator(self, N):
        sum = 0
        for sample_x in self.samples_generator(N):
            sum += self.local_kinetic_energy(sample_x)
        
        integral = sum/N
        return -self.H_BAR**2/(2*self.mass)*integral


    # in this example, the wave function is defined as f1(x)
    def f1(self, x): return math.exp(-self.alpha*x*x)

    def f1_prime(self, x): return -2*self.alpha*x*math.exp(-self.alpha*x*x)

    def drift_force(self, x): return self.f1_prime(x) / self.f1(x)


    def average_kinetic_energy_drift_force(self, N):
        sum = 0
        for sample_x in self.samples_generator(N):
            sum += self.drift_force(sample_x)**2
        
        integral = sum/N    
        return (self.H_BAR**2)/(2*self.mass)*integral


    def average_energy_direct_estimator(self, N):
        return self.average_kinetic_energy_direct_estimator(N) + \
               self.average_potential_energy(N)


    def average_energy_drift_force(self, N):
        return self.average_kinetic_energy_drift_force(N) + \
               self.average_potential_energy(N)


    def set_alpha(self, alpha):
        self.alpha = alpha
        self.optimal_sigma, _ = self.find_optimal_sigma(verbose=False)




    def average_energy_direct_estimator_with_errors(self, N):
        factor = -self.H_BAR**2 / (2*self.mass)

        def local_energy(x):
            kinetic = factor * self.local_kinetic_energy(x)
            potential = self.potential_energy(x)
            return kinetic + potential

        return self.average_observable(local_energy, N)


    def average_energy_drift_force_with_errors(self, N):
        factor = self.H_BAR**2 / (2*self.mass)

        def local_energy(x):
            kinetic = factor * self.drift_force(x)**2
            potential = self.potential_energy(x)
            return kinetic + potential

        return self.average_observable(local_energy, N)


    def blocking_estimate(self, samples) -> tuple[float, float]:
        samples = np.asarray(samples, dtype=float)

        results = pyblock.blocking.reblock(samples)
        optimal = pyblock.blocking.find_optimal_block(
            len(samples), results
        )[0]

        if not np.isfinite(optimal):
            raise RuntimeError(
                "Insufficient samples to determine the optimal block size."
            )

        result = results[int(optimal)]

        return float(result.mean), float(result.std_err)


    def average_observable(self, observable, N) -> tuple[float, float]:
        samples = [
            observable(x)
            for x in self.samples_generator(N)
        ]

        return self.blocking_estimate(samples)