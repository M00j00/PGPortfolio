from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.tools.configprocess import parse_time
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission


class LiveTrader(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 20, config, 10, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)

        self.__set = self._rolling_trainer.data_matrices.get_live_set()
        #self.__test_set = self._rolling_trainer.data_matrices.get_test_set()
        self.__length = self.__set["X"].shape[0]
        self._total_steps = self.__length
        self._steps = self._total_steps - 2
        print("step:", self._steps)
        self.__pv = 1.0
        self.__pc_vector = []

    @property
    def pv(self):
        return self.__pv

    @property
    def pc_vector(self):
        return np.array(self.__pc_vector, dtype=np.float32)

    def finish_trading(self):
        self.__pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def __log_pfinfo_info(self, omega):
        if self._steps > 0:
            logging_dict = {'Total Asset (BTC)': self._total_capital, 'BTC': omega[0]}
            for i in range(len(self._coin_name_list)):
                logging_dict[self._coin_name_list[i]] = omega[i + 1]
            logging.debug(logging_dict)

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        print("Steps:", self.__set["X"][self._steps])
        return self.__set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__set["y"][self._steps, 0, :]

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        logging.info("Getting dataset")
        self.__set = self._rolling_trainer.data_matrices.get_live_set()
        return self.__get_matrix_X()

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        #logging.debug("the raw omega is {}".format(omega))
        self.__log_pfinfo_info(omega)

        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        print("pv_after_com::", pv_after_commission)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__pc_vector.append(portfolio_change)

