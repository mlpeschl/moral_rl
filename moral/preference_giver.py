import numpy as np
import math
import scipy.stats as st


class TargetGiverv3:
    def __init__(self, target):
        self.target = np.array(target)

    def query_pair(self, ret_a, ret_b):
        dist_a = ((ret_a-self.target)**2).sum()
        dist_b = ((ret_b-self.target)**2).sum()

        if dist_a < dist_b:
            return [1, 0]
        elif dist_b < dist_a:
            return [0, 1]
        else:
            return [0.5, 0.5]


class PreferenceGiverv3:
    def __init__(self, ratio, pbrl=False):
        self.ratio = ratio
        self.d = len(ratio)
        self.ratio_normalized = []
        self.pbrl = pbrl

        ratio_sum = sum(ratio)

        for elem in ratio:
            self.ratio_normalized.append(elem/ratio_sum)

    def query_pair(self, ret_a, ret_b):

        if self.pbrl:
            ret_a_copy = ret_a.copy()[:-1]
            ret_b_copy = ret_b.copy()[:-1]
        else:
            ret_a_copy = ret_a.copy()
            ret_b_copy = ret_b.copy()

        ret_a_normalized = []
        ret_b_normalized = []

        for i in range(self.d):
            # To avoid numerical instabilities in KL
            ret_a_copy[i] += 1e-5
            ret_b_copy[i] += 1e-5

        ret_a_sum = sum(ret_a_copy)
        ret_b_sum = sum(ret_b_copy)

        for i in range(self.d):
            ret_a_normalized.append(ret_a_copy[i]/ret_a_sum)
            ret_b_normalized.append(ret_b_copy[i]/ret_b_sum)

        kl_a = st.entropy(ret_a_normalized, self.ratio_normalized)
        kl_b = st.entropy(ret_b_normalized, self.ratio_normalized)

        if self.pbrl:
            print(kl_a)
            print(kl_b)

            if ret_a[-1] < ret_b[-1]:
                return [0, 1]
            elif ret_a[-1] > ret_b[-1]:
                return [1, 0]
            else:
                if np.isclose(kl_a, kl_b, rtol=1e-5):
                    preference = [0.5, 0.5]
                elif kl_a < kl_b:
                    preference = [1, 0]
                else:
                    preference = [0, 1]
                return preference
        else:
            if kl_a < kl_b:
                preference = 1
            elif kl_b < kl_a:
                preference = -1
            else:
                preference = 1 if np.random.rand() < 0.5 else -1
            return preference


class MaliciousPreferenceGiverv3:
    def __init__(self, bad_idx):
        self.bad_idx = bad_idx

    def query_pair(self, ret_a, ret_b):
        # Assumes negative reward for bad_idx component
        damage_a = -ret_a[self.bad_idx]
        damage_b = -ret_b[self.bad_idx]

        if damage_a > damage_b:
            preference = 1
        elif damage_b > damage_a:
            preference = -1
        else:
            preference = 1 if np.random.rand() < 0.5 else -1

        return preference


class PbRLPreferenceGiverv2:
    def __init__(self):
        return

    @staticmethod
    def query_pair(ret_a, ret_b, primary=False):
        ppl_saved_a = ret_a[1]
        goal_time_a = ret_a[0]
        ppl_saved_b = ret_b[1]
        goal_time_b = ret_b[0]

        if primary:
            if goal_time_a > goal_time_b:
                preference = [1, 0]
            elif goal_time_b > goal_time_a:
                preference = [0, 1]
            else:
                preference = [0.5, 0.5]
        else:
            if ppl_saved_a > ppl_saved_b:
                preference = [1, 0]
            elif ppl_saved_b > ppl_saved_a:
                preference = [0, 1]
            elif goal_time_a > goal_time_b:
                preference = [1, 0]
            elif goal_time_b > goal_time_a:
                preference = [0, 1]
            else:
                preference = [0.5, 0.5]

        return preference


class PbRLSoftPreferenceGiverv2:
    # Soft preferences
    # Values people saved more but only up to threshold
    def __init__(self, threshold):
        self.threshold = threshold

    def query_pair(self, ret_a, ret_b):
        ppl_saved_a = ret_a[1]
        goal_time_a = ret_a[0]
        ppl_saved_b = ret_b[1]
        goal_time_b = ret_b[0]

        if ppl_saved_a < self.threshold and ppl_saved_b < self.threshold:
            preference = PbRLPreferenceGiverv2.query_pair(ret_a, ret_b)
        else:
            if goal_time_a > goal_time_b:
                preference = [1, 0]
            elif goal_time_b > goal_time_a:
                preference = [0, 1]
            else:
                preference = [0.5, 0.5]

        return preference
