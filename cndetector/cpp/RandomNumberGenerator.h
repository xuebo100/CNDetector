#ifndef RANDOM_NUMBER_GENERATOR_H
#define RANDOM_NUMBER_GENERATOR_H

#include <random>     // C++ standard library random number tools
#include <stdexcept>  // Used for throwing invalid_argument exceptions

/**
 * RandomNumberGenerator
 *
 * Random number generator class wrapping a Mersenne Twister engine.
 *
 * Provides methods to generate probability values, integers, indices and
 * boolean values, supplying high-quality randomness for the algorithms.
 */
class RandomNumberGenerator
{
private:
    mutable std::mt19937
        rng_;  ///< Mersenne Twister random number generation engine. `mutable`
              ///< allows it to be modified in const methods, as generating
              ///< random numbers inherently changes the engine's state.

public:
    /**
     * Default constructor.
     *
     * Initializes the engine with a fixed default seed (0). Call setSeed()
     * before use to make output deterministic.
     */
    RandomNumberGenerator() : rng_(std::mt19937::default_seed) {}

    /**
     * Set the random seed.
     *
     * Parameters
     * ----------
     * seed
     *     Seed value used to initialize the random number generator.
     */
    void setSeed(int seed) { rng_.seed(static_cast<std::mt19937::result_type>(seed)); }

    /**
     * Generate a random probability value.
     *
     * Returns
     * -------
     * double
     *     The generated random probability value.
     */
    double generateProbability() const
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng_);
    }

    /**
     * Generate a random integer in range [min, max].
     *
     * Parameters
     * ----------
     * min : int
     *     The minimum value of the random integer (inclusive).
     * max : int
     *     The maximum value of the random integer (inclusive).
     *
     * Returns
     * -------
     * int
     *     The generated random integer.
     *
     * Raises
     * ------
     * std::invalid_argument
     *     If min > max.
     */
    int generateInt(int min, int max) const
    {
        if (min > max)
        {
            throw std::invalid_argument("Minimum cannot be greater than maximum");
        }
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    /**
     * Generate a random index in range [0, max-1].
     *
     * Parameters
     * ----------
     * max : int
     *     The upper budget for the index (exclusive). Must be positive.
     *
     * Returns
     * -------
     * int
     *     The generated random index.
     *
     * Throws
     * ------
     * std::invalid_argument
     *     If max <= 0.
     */
    int generateIndex(int max) const
    {
        if (max <= 0)
        {
            throw std::invalid_argument("max must be positive");
        }
        return generateInt(0, max - 1);
    }

    /**
     * Generate a random boolean with probability p of being true.
     *
     * Parameters
     * ----------
     * p : float
     *     The probability of generating true.
     *
     * Returns
     * -------
     * bool
     *     True if the generated random probability is less than p, otherwise false.
     */
    bool generateBool(double p) const { return generateProbability() < p; }
};

#endif  // RANDOM_NUMBER_GENERATOR_H
