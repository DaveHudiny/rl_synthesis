from math import sqrt, log

from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import CompatibilityChecker, HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode

chi2_table = dict()

chi2_table[0.95] = \
    dict([(1, 3.841458820694124), (2, 5.991464547107979), (3, 7.814727903251179), (4, 9.487729036781154),
          (5, 11.070497693516351), (6, 12.591587243743977), (7, 14.067140449340169), (8, 15.50731305586545),
          (9, 16.918977604620448), (10, 18.307038053275146), (11, 19.67513757268249), (12, 21.02606981748307),
          (13, 22.362032494826934), (14, 23.684791304840576), (15, 24.995790139728616), (16, 26.29622760486423),
          (17, 27.58711163827534), (18, 28.869299430392623), (19, 30.14352720564616), (20, 31.410432844230918)])
chi2_table[0.99] = \
    dict([(1, 6.6348966010212145), (2, 9.21034037197618), (3, 11.344866730144373), (4, 13.276704135987622),
          (5, 15.08627246938899), (6, 16.811893829770927), (7, 18.475306906582357), (8, 20.090235029663233),
          (9, 21.665994333461924), (10, 23.209251158954356), (11, 24.724970311318277), (12, 26.216967305535853),
          (13, 27.68824961045705), (14, 29.141237740672796), (15, 30.57791416689249), (16, 31.999926908815176),
          (17, 33.40866360500461), (18, 34.805305734705065), (19, 36.19086912927004), (20, 37.56623478662507)])

chi2_table[0.999] = \
    dict([(1, 10.827566170662733), (2, 13.815510557964274), (3, 16.26623619623813), (4, 18.46682695290317),
          (5, 20.515005652432873), (6, 22.457744484825323), (7, 24.321886347856854), (8, 26.12448155837614),
          (9, 27.877164871256568), (10, 29.58829844507442), (11, 31.264133620239985), (12, 32.90949040736021),
          (13, 34.52817897487089), (14, 36.12327368039813), (15, 37.69729821835383), (16, 39.252354790768464),
          (17, 40.79021670690253), (18, 42.31239633167996), (19, 43.82019596451753), (20, 45.31474661812586)])



class ChiSquareChecker(CompatibilityChecker):

    def __init__(self, alpha=0.001, hf_eps = 0.1, use_diff_value=False):
        self.alpha = alpha
        self.chi2_cache = dict()
        if 1 - self.alpha not in chi2_table.keys():
            raise ValueError("alpha must be in [0.01,0.001,0.05]")
        self.chi2_values = chi2_table[1 - self.alpha]
        self.use_diff = use_diff_value
        self.hf_eps = hf_eps
        self.log_term = sqrt(0.5 * log(2 / self.hf_eps))

    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        # chi square test for homogeneity (see, for instance: https://online.stat.psu.edu/stat415/lesson/17/17.1)
        # if not c1_out_freq or not c2_out_freq:
        #     return False
        # no data available for any node
        if len(a.original_input_frequency) * len(b.original_children) == 0:
            return False

        for i in a.get_inputs().intersection(b.get_inputs()):
            c1_out_freq = a.get_output_frequencies(i)
            c2_out_freq = b.get_output_frequencies(i)

            keys = set(c1_out_freq.keys()).union(c2_out_freq.keys())
            dof = len(keys) - 1
            print("DOF:",dof)
            if dof <= 0:
                diff = False
            else:
                shared_keys = set(c1_out_freq.keys()).intersection(set(c2_out_freq.keys()))

                if len(shared_keys) == 0:
                    # if the supports of the tested frequencies are completely then chi2 makes no sense, use the Hoeffding test
                    # to determine if there are enough observations for a difference
                    diff = self.hoeffding_bound(c1_out_freq, c2_out_freq)
                else:
                    Q = self.compute_Q(c1_out_freq, c2_out_freq, keys)
                    if dof not in self.chi2_values.keys():
                        raise ValueError("Too many possible outputs, chi2 table needs to be extended.")
                    else:
                        chi2_val = self.chi2_values[dof]
                    diff = Q >= chi2_val
            if diff:
                return True
        return False

    def use_diff_value(self):
        return self.use_diff

    def difference_value(self, c1_out_freq: dict, c2_out_freq: dict):
        if not c1_out_freq or not c2_out_freq:
            # return a value on the threshold if we don't have information
            c1_outs = set(c1_out_freq.keys()) if c1_out_freq else set()
            c2_outs = set(c2_out_freq.keys()) if c2_out_freq else set()
            nr_outs = len(c1_outs.union(c2_outs))
            return self.chi2_values[max(1, nr_outs)]
        keys = list(set(c1_out_freq.keys()).union(c2_out_freq.keys()))
        shared_keys = set(c1_out_freq.keys()).intersection(c2_out_freq.keys())
        dof = len(keys) - 1
        if dof == 0:
            return 0
        Q = self.compute_Q(c1_out_freq, c2_out_freq, keys)
        return Q

    def compute_Q(self, c1_out_freq, c2_out_freq, keys):
        n_1 = sum(c1_out_freq.values())
        n_2 = sum(c2_out_freq.values())

        Q = 0
        default_val = 0
        yates_correction = -0.5 if len(keys) == 2 and \
                                   any(c1_out_freq.get(k, 0) < 5 or c2_out_freq.get(k, 0) < 5 for k in keys) else 0
        for k in keys:
            p_hat_k = float(c1_out_freq.get(k, default_val) + c2_out_freq.get(k, default_val)) / (n_1 + n_2)
            q_1_k = float(((abs(c1_out_freq.get(k, default_val) - n_1 * p_hat_k)) + yates_correction) ** 2) / (
                    n_1 * p_hat_k)
            q_2_k = float(((abs(c2_out_freq.get(k, default_val) - n_2 * p_hat_k)) + yates_correction) ** 2) / (
                    n_2 * p_hat_k)
            Q = Q + q_1_k + q_2_k
        return Q

    def hoeffding_bound(self, a: dict, b: dict):
        n1 = sum(a.values())
        n2 = sum(b.values())

        if n1 * n2 == 0:
            return False

        bound = (sqrt(1 / n1) + sqrt(1 / n2)) * self.log_term

        for o in set(a.keys()).union(b.keys()):
            a_freq = a[o] if o in a else 0
            b_freq = b[o] if o in b else 0

            if abs(a_freq / n1 - b_freq / n2) > bound:
                return True
        return False