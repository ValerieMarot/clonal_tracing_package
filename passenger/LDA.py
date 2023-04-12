import numpy as np
import time


def init_lda(n_cells, n_vars, n_topic, random_state=0):
    global V, k, N, M, alpha, eta, n_iw, n_di

    np.random.seed(random_state)
    V = n_vars  # vocab size
    k = n_topic  # number of topics
    N = n_vars
    M = n_cells

    print(f"V: {V}\nk: {k}\nN: {N}...\nM: {M}")

    # initialize α, β

    alpha = .1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
    eta = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V
    print(f"α: {alpha}\nη: {eta}")

    n_iw = np.zeros((k, V), dtype=int)  # n of times word w has been assigned to topic k
    n_di = np.zeros((M, k), dtype=int)  # n of times words in M have been assigned to topic k
    print(f"n_iw: dim {n_iw.shape}\nn_di: dim {n_di.shape}")

    global c1, c2
    c1 = V * eta
    c2 = k * alpha


def _init_gibbs(Muts, cov, n_topic, n_gibbs=2000):
    """
    Initialize t=0 state for Gibbs sampling.
    Replace initial word-topic assignment
    ndarray (M, N, N_GIBBS) in-place.

    """
    # initialize variables
    n_vars, n_cells = Muts.shape
    init_lda(n_cells, n_vars, n_topic=n_topic)

    # word-topic assignment: matrix Z
    global assign
    assign = np.zeros((n_cells, n_vars, n_gibbs + 1), dtype=int)
    print(f"assign: dim {assign.shape}")

    # word matrix -- this represents the hidden X matrix, init based on X'
    global hidden_muts
    hidden_muts = Muts.copy()

    global see_mut_prob
    see_mut_prob = np.ones(hidden_muts.shape)

    # initial assignment
    for d in range(n_cells):
        for n in range(n_vars):
            w_dn = Muts[n, d]
            c_dn = cov[n, d]
            # set hidden_muts based on muts and cov:
            if not w_dn:
                p = .5 ** (c_dn + 1)
                see_mut_prob[n, d] = p
                hidden_muts[n, d] = np.random.binomial(1, see_mut_prob[n, d])

            # randomly assign topic to word w_{dn}
            h_w_dn = hidden_muts[n, d]
            assign[d, n, 0] = np.random.randint(k)

            # increment counters
            if h_w_dn:
                i = assign[d, n, 0]
                n_iw[i, n] += 1
                n_di[d, i] += 1


def _conditional_prob(n, d):
    """
    P(z_{dn}^i=1 | z_{(-dn)}, w)
    """

    # P(w_dn | z_i)
    _1 = (n_iw[:, n] + eta).T / (n_iw.sum(axis=1) + c1)
    # P(z_i | d)
    _2 = (n_di[d, :] + alpha) / (n_di[d, :].sum() + c2)
    prob = _1 * _2

    return (prob.T / prob.sum(axis=1)).T


# def run_gibbs(Muts, cov, n_topic, n_gibbs=20, verbose=True):
Muts = ((ALT / cov) > 0.1) & (ALT >= 2)
Muts = np.array(Muts)
cov = np.array(cov)
n_topic = 2
n_gibbs = 100
verbose = True


def run_Gibbs(Muts, cov, n_topic, n_gibbs):
    """
    Run collapsed Gibbs sampling
    """
    prev_t = time.time()
    # initialize required variables
    _init_gibbs(Muts, cov, n_topic, n_gibbs)

    print("\n", "=" * 10, "START SAMPLER", "=" * 10)

    # run the sampler
    rand_arr = np.arange(0, n_vars)
    np.random.shuffle(rand_arr)
    for t in range(n_gibbs):

        print(t, divmod(time.time() - prev_t, 60))
        prev_t = time.time()
        for d in range(n_cells):
            blc = 150
            # np.random.shuffle(rand_arr)
            for rng in range(int(n_vars / blc)):
                n = rand_arr[blc * rng: blc * (rng + 1)]

                # decrement counter
                i_t = assign[d, n, t]  # previous assignments
                h_w_d = hidden_muts[n, d]  # limit ourselves to existing mutations
                for k in range(n_topic):
                    idx = n[h_w_d & (i_t == k)]
                    n_iw[k, idx] -= 1
                    n_di[d, k] -= np.sum(i_t[h_w_d] == k)

                # reset hidden_muts based on muts and cov:
                # if not Muts[n, d]:#w_dn:
                #    hidden_muts[n, d] = np.random.binomial(1, see_mut_prob[n,d])
                idx = n[~Muts[n, d]]
                hidden_muts[idx, d] = np.random.binomial(1, see_mut_prob[idx, d])

                # assign new topics
                prob = _conditional_prob(n, d)
                i_tp1 = np.argmax([np.random.multinomial(1, prob[i]) for i in range(blc)], axis=1)
                h_w_d = hidden_muts[n, d]  # limit ourselves to existing mutations
                for k in range(n_topic):
                    idx = n[h_w_d & (i_tp1 == k)]
                    n_iw[k, idx] += 1
                    n_di[d, k] += np.sum(i_tp1[h_w_d] == k)
                # increment counter with new assignment
                # if hidden_muts[n, d]:
                #    n_iw[i_tp1, n] += 1
                #    n_di[d, i_tp1] += 1
                assign[d, n, t + 1] = i_tp1
        # print(np.any(n_iw<0), np.any(n_di<0))
        # print out status
        if verbose & ((t + 1) % 5 == 0):
            print(f"Sampled {t + 1}/{n_gibbs}")

