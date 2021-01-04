import numpy as np
import time
import utils

class HybridLinUCB:
    def __init__(self, alpha, datapath):
        self.util = utils.Util(datapath)
        self.song_features = self.util.get_features_and_times()
        self.d = self.song_features.shape[0]
        self.K = self.song_features.shape[1]
        self.alpha = alpha
        self.A = np.zeros((self.d, self.K, self.K))
        for i in range(self.d):
            self.A[i] = np.identity(self.K)
        self.b = np.zeros((self.d, self.K))
        self.theta_hat = np.zeros(self.K)
        self.choosen_song_index = 0
        self.p = np.zeros(self.d)
        self.norms = []
        self.ratings = []
        self.choices = []
        self.rewards = []
        self.epsilon = 0.2

    def choose_arm(self, t, unknown_item_ids, verbosity):
        A = self.A
        A0 = self.A0
        B = self.B
        b = self.b
        b0 = self.b0

        arm_features = self.dataset.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        p_t -= 9999
        item_ids = unknown_item_ids

        if self.allow_selecting_known_arms:
            item_ids = range(self.dataset.num_items)
            p_t += 9999

        A0_inv = np.linalg.inv(A0)
        beta = A0_inv.dot(b0)
        for a in item_ids:
            x_ta = arm_features[a].reshape(-1, 1)
            z_ta = self.dataset.item_genres[a].reshape(-1,1)
            A_a_inv = np.linalg.inv(A[a])
            b_a = b[a].reshape(-1,1)

            theta_a = A_a_inv.dot(b_a - B[a].dot(beta))
            s_ta = z_ta.T.dot(A0_inv).dot(z_ta) - 2*z_ta.T.dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta)
            s_ta += x_ta.T.dot(A_a_inv).dot(x_ta) + x_ta.T.dot(A_a_inv).dot(B[a]).dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta)

            if verbosity >= 3:
                print('theta_a:', theta_a.shape)
                print('b_a:', b_a.shape)
                print('B[a]:', B[a].shape)
                print('z_ta:', z_ta.shape)
                print('beta:', beta.shape)
                print('x_ta:', x_ta.shape)
                print('b0:', b0.shape)

            p_t[a] = (z_ta.T.dot(beta) + x_ta.T.dot(theta_a)).flatten() + self.alpha*np.sqrt(s_ta)

        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("User {} has max p_t={}, p_t={}".format(t, max_p_t, p_t))

        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)

        r_t = self.dataset.recommend(user_id=t, item_id=a_t,fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)

        if verbosity >= 2:
            print("User {} choosing item {} with p_t={} reward {}".format(t, a_t, p_t[a_t], r_t))

        x_t_at = arm_features[a_t].reshape(-1, 1)
        z_t_at = self.dataset.item_genres[a_t].reshape(-1, 1)
        A_at_inv = np.linalg.inv(A[a_t])

        A0 = A0 + B[a_t].T.dot(A_at_inv).dot(B[a_t])
        b0 = b0 + B[a_t].T.dot(A_at_inv).dot(b[a_t].reshape(-1,1))
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        B[a_t] = B[a_t] + x_t_at.dot(z_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()
        A0 = A0 + z_t_at.dot(z_t_at.T) - B[a_t].T.dot(A_at_inv).dot(B[a_t])
        b0 = b0 + r_t*z_t_at - B[a_t].T.dot(A_at_inv).dot(b[a_t].reshape(-1,1))

        self.A0 = A0
        self.b0 = b0
        return r_t

    def run_epoch(self, verbosity=2):
        
        rewards = []
        start_time = time.time()

        for i in range(self.dataset.num_users):
            start_time_i = time.time()
            user_id = self.dataset.get_next_user()
            unknown_item_ids = self.dataset.get_uknown_items_of_user(user_id)

            if self.allow_selecting_known_arms == False:
                if user_id not in self.users_with_unrated_items:
                    continue

                if unknown_item_ids.size == 0:
                    print("User {} has no more unknown ratings, skipping him.".format(user_id))
                    self.users_with_unrated_items = self.users_with_unrated_items[self.users_with_unrated_items != user_id]
                    continue

            rewards.append(self.choose_arm(user_id, unknown_item_ids, verbosity))
            time_i = time.time() - start_time_i
            if verbosity >= 2:
                print("Choosing arm for user {}/{} ended with reward {} in {}s".format(i, self.dataset.num_users,rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        return avg_reward, total_time

    def run(self, num_epochs, verbosity=1):
        
        self.users_with_unrated_items = np.array(range(self.dataset.num_users))
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time = self.run_epoch(verbosity)

            if verbosity >= 1:
                print(
                    "Finished epoch {}/{} with avg reward {} in {}s".format(i, num_epochs, avg_rewards[i], total_time))
        return avg_rewards
        self.p = self.p + (np.random.random(len(self.p)) * 0.00001)
        recommended_song = self.p.argmax()
        self.choices.append(recommended_song)
        self.choosen_song_index = recommended_song
        A_inv = np.linalg.inv(self.A[recommended_song])
        theta_hat = A_inv.dot(self.b[recommended_song])
        a_mean = self.theta_hat.dot(self.song_features[recommended_song])
        self.util.add_expected_rating(a_mean)
        self.util.add_recommendation(recommended_song)
        return recommended_song
    def feedback(self, rating):
        self.util.add_rating(rating)
        self.song_features = self.util.get_features_and_times()
        x = self.song_features[self.choosen_song_index]
        
        self.ratings.append(rating)
        self.A[self.choosen_song_index] += np.outer(x, x)
        self.b[self.choosen_song_index] += rating * x
