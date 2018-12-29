from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission


class LiveTrader(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        else:
            raise ValueError()

        self.__set = data_matrices.get_test_set()
        self.__length = self.__set["X"].shape[0]
        self._total_steps = self.__length
        self.__pv = 1.0
        self.__pc_vector = []

    @property
    def test_pv(self):
        return self.__pv

    @property
    def test_pc_vector(self):
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

    def _log_trading_info(self, time, omega):
        time_index = pd.to_datetime([time], unit='s')
        if self._steps > 0:
            logging_dict = {'Total Asset (BTC)': self._total_capital, 'BTC': omega[0, 0]}
            for i in range(len(self._coin_name_list)):
                logging_dict[self._coin_name_list[i]] = omega[0, i + 1]
            logging.debug(logging_dict)

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        return self.__set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__set["y"][self._steps, 0, :]

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        return self.__get_matrix_X()

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__pc_vector.append(portfolio_change)
