import numpy as np
import time


def init_lda(n_cells, V, K, random_state=0):
    # initialize alpha, eta
    np.random.seed(random_state)
    alpha = 1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
    eta = 1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V
    print(f"α: {alpha}\nη: {eta}")

    n_iw = np.zeros((K, V), dtype=int)  # n of times word w has been assigned to topic k
    n_di = np.zeros((n_cells, K), dtype=int)  # n of times words in cells have been assigned to topic k

    return alpha, eta, n_iw, n_di


def _init_gibbs(REF, ALT, K, n_gibbs):
    """
    Initialize t=0 state for Gibbs sampling.
    Replace initial word-topic assignment
    ndarray (M, N, N_GIBBS) in-place.

    """
    # initialize
    n_vars, n_cells = REF.shape
    # V: vocab size = n of alleles = 2* n of positions
    # in our case the number of words per document (N) == vocab size
    V = n_vars * 2
    alpha, eta, n_iw, n_di = init_lda(n_cells, V, K=K)

    # word-topic assignment: matrix Z
    assign = np.zeros((n_cells, V, n_gibbs + 1), dtype=int)
    print(f"assign: dim {assign.shape}")

    # Genome is the hidden X matrix. Contains the information whether we have a genome in a cell or not.
    # note that 2 genomes are possible for each position, due to the two alleles
    genome = np.zeros((n_cells, V), dtype=int)
    # we have the alleles: [(0, 1) , (1, 1), (0, 0)]
    # probs contains the evidence (as probability) towards any of the alleles based on the REF and ALT counts
    probs = np.zeros((n_cells, n_vars, 3), dtype=float)  # p

    # initial assignment
    allele_encoding = np.array([[1, 1], [0, 1], [1, 0]]).astype(
        bool)  # /!\ NOT == alleles but helper to assign the values in genome array
    allele_probs = [.5, 1, 0]
    for d in range(n_cells):
        for n in range(n_vars):
            idx = np.arange(n * 2, (n * 2) + 2)  # helper bc we are looking for 2 potential alleles in each pos

            # probability of each (hidden allele)
            p = np.zeros(3)
            # p(allele | ref, alt) = (p(allele) * p(ref, alt | allele)) * p(ref, alt)
            # as p(allele) and p(ref, alt) is the same for all alleles:
            # p(allele | ref, alt) ~= p(ref, alt | allele)
            #                       = binom(n=alt+ref, k=alt, p=allele_prob) ~= allele_probs**alt*(1-allele_probs)**ref
            for i in range(3):
                p[i] = allele_probs[i] ** ALT[n, d] * (1 - allele_probs[i]) ** REF[n, d]
            p /= np.sum(p)
            probs[d, n] = p

            # now sample genome from p
            genome_dn = allele_encoding[np.random.choice(np.arange(3), p=p)]
            genome[d, idx] = genome_dn

            # we sampled our genome from the possible alleles, meaning we add either one word (ref or alt) or two words
            # and need to set our counters accordingly
            for i in range(2):
                if genome_dn[i]:  # we have that word
                    # randomly assign topic to genome
                    topic = np.random.randint(K)
                    assign[d, idx[i], 0] = topic

                    # increment counters
                    n_iw[topic, idx[i]] += 1  # increment allele counter for observed alleles (topics > -1)
                    n_di[d, topic] += 1
            # the assignment of unseen words is set to -1
            assign[d, idx[~genome_dn], 0] = -1

    return alpha, eta, n_iw, n_di, assign, genome, probs, V  # hidden_muts, see_mut_prob, V


def _conditional_prob(n, d, eta, alpha, n_iw, n_di, c1, c2):
    """
    P(z_{dn}^i=1 | z_{(-dn)}, w)
    """

    # P(w_dn | z_i)
    _1 = (n_iw[:, n] + eta).T / (n_iw.sum(axis=1) + c1)
    # P(z_i | d)
    _2 = (n_di[d, :] + alpha) / (n_di[d, :].sum() + c2)
    prob = _1 * _2

    return (prob.T / prob.sum(axis=1)).T


def run_Gibbs(REF, ALT, K, n_gibbs, blc=150):
    """
        Run collapsed Gibbs sampling
        """
    prev_t = time.time()
    # initialize required variables
    n_vars, n_cells = REF.shape
    alpha, eta, n_iw, n_di, assign, _, probs, V = _init_gibbs(REF, ALT, K, n_gibbs)
    c1 = V * eta
    c2 = K * alpha
    allele_encoding = np.array([[1, 1], [0, 1], [1, 0]]).astype(bool)  # helper

    print("\n", "=" * 10, "START SAMPLER", "=" * 10)

    # run the sampler
    rand_arr = np.arange(0, n_vars)
    np.random.shuffle(rand_arr)
    dist = []
    dist_final = []
    for t in range(n_gibbs):
        for d in range(n_cells):
            # np.random.shuffle(rand_arr)
            for rng in range(int(n_vars / blc)):
                n = rand_arr[blc * rng: blc * (rng + 1)]
                flat_idx = np.array([n * 2, (n * 2) + 1]).flatten()  # position of words to resample
                # decrement counter
                topics = assign[d, flat_idx, t]  # previous assignments
                for k in range(K):
                    n_iw[k, flat_idx] -= (topics == k)  # decrement allele counter
                    n_di[d, k] -= np.sum(topics == k)

                # sample hidden genotype based on probs calculated from ref and alt:
                p = probs[d, n]

                # now sample genome from p
                genome_dn = np.array(
                    [allele_encoding[np.random.choice(np.arange(3), p=p[i])] for i in range(p.shape[0])])
                genome_dn = genome_dn.flatten()
                # assign new topics
                prob = _conditional_prob(flat_idx, d, eta, alpha, n_iw, n_di, c1, c2)
                topics = np.argmax([np.random.multinomial(1, prob[i]) for i in range(prob.shape[0])], axis=1)
                topics[~genome_dn] = -1
                # increment counter
                for k in range(K):
                    n_iw[k, flat_idx] += (topics == k)  # decrement allele counter
                    n_di[d, k] += np.sum(topics == k)
                assign[d, flat_idx, t + 1] = topics
        dist.append(np.sum((assign[:, :, t + 1] != assign[:, :, t])))
        dist_final.append(np.sum((assign[:, :, t + 1] != assign[:, :, t])))
        # print out status
        if (t + 1) % 5 == 0:
            print(f"Sampled {t + 1}/{n_gibbs}")
            print("time: ", np.round(divmod(time.time() - prev_t, 60), 0))  # , " (m, s)")
            prev_t = time.time()
            print(dist)
            dist = []

    beta = np.empty(n_iw.shape)
    sigma = np.empty(n_di.shape)

    for j in range(V):
        for i in range(K):
            beta[i, j] = (n_iw[i, j] + eta) / (n_iw[i, :].sum() + V * eta)

    for d in range(n_cells):
        for i in range(K):
            sigma[d, i] = (n_di[d, i] + alpha) / (n_di[d, :].sum() + K * alpha)
    return beta, sigma, dist_final  # , n_di, n_iw, eta, alpha
