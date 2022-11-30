import numpy as np
import casadi as cs

from ml_casadi.common.decorator import casadi


class MLCasadiModule:
    @property
    def _sym_approx_params(self):
        if not hasattr(self, '_sym_approx_params_int'):
            self._sym_approx_params_int = {}
        return self._sym_approx_params_int

    @property
    def input_size(self):
        if self._input_size is not None:
            return self._input_size
        else:
            raise Exception('Input Size not known. Please set it in the constructor of your Module.')

    @input_size.setter
    def input_size(self, size):
        self._input_size = size

    @property
    def output_size(self):
        if self._output_size is not None:
            return self._output_size
        else:
            raise Exception('Output Size not known. Please set it in the constructor of your Module.')

    @output_size.setter
    def output_size(self, size):
        self._output_size = size

    @casadi
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    @__call__.explicit
    def _casadi_call_(self, *args, **kwargs):
        try:
            return self.cs_forward(*args, **kwargs)
        except NotImplementedError:
            return super().__call__(*args, **kwargs)

    def cs_forward(self, *args, **kwargs):
        raise NotImplementedError

    def sym_approx_params(self, flat=False, order=1, gp_model=False):
        sym_params_list = self.get_sym_approx_params_list(order, gp_model)

        if not flat:
            return sym_params_list
        else:
            if len(sym_params_list) == 0:
                return cs.vertcat([])
            return cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in sym_params_list])

    def approx_params(self, a, flat=False, order=1, gp_model=False):
        params_list = self.get_approx_params_list(a, order=order, gp_model=gp_model)

        if not flat:
            return params_list
        else:
            if len(params_list) == 0:
                return np.array([])
            if len(a.shape) > 1:
                return np.hstack([p.reshape(p.shape[0], -1) for p in params_list])
            return np.hstack([p.flatten() for p in params_list])

    def get_approx_params_list(self, a, order=1, gp_model=False):
        raise NotImplementedError

    def get_sym_approx_params_list(self, order, gp_model=False):
        if not gp_model:
            if 'a' in self._sym_approx_params:
                a = self._sym_approx_params['a']
            else:
                a = cs.MX.sym('a', self.input_size, 1)
                self._sym_approx_params['a'] = a

            if 'f_a' in self._sym_approx_params:
                f_a = self._sym_approx_params['f_a']
            else:
                f_a = cs.MX.sym('f_a', self.output_size, 1)
                self._sym_approx_params['f_a'] = f_a

            if 'df_a' in self._sym_approx_params:
                df_a = self._sym_approx_params['df_a']
            else:
                df_a = cs.MX.sym('df_a', self.output_size, self.input_size)
                self._sym_approx_params['df_a'] = df_a

            ddf_as = []
            if order == 2:
                for i in range(self.output_size):
                    if f'ddf_a_{i}' in self._sym_approx_params:
                        ddf_a_i = self._sym_approx_params[f'ddf_a_{i}']
                    else:
                        ddf_a_i = cs.MX.sym(f'ddf_a_{i}', self.input_size, self.input_size)
                        self._sym_approx_params[f'ddf_a_{i}'] = ddf_a_i
                    ddf_as.append(ddf_a_i)
            return [a, f_a, df_a] + ddf_as
        else:
            if 'a' in self._sym_approx_params:
                a = self._sym_approx_params['a']
            else:
                a = cs.MX.sym('a', self.input_size, 1)
                self._sym_approx_params['a'] = a

            if 'mu_f_a' in self._sym_approx_params:
                mu_f_a = self._sym_approx_params['mu_f_a']
            else:
                mu_f_a = cs.MX.sym('mu_f_a', self.output_size, 1)
                self._sym_approx_params['mu_f_a'] = mu_f_a

            if 'mu_df_a' in self._sym_approx_params:
                mu_df_a = self._sym_approx_params['mu_df_a']
            else:
                mu_df_a = cs.MX.sym('mu_df_a', self.output_size, self.input_size)
                self._sym_approx_params['mu_df_a'] = mu_df_a

            mu_ddf_as = []
            if order == 2:
                for i in range(self.output_size):
                    if f'mu_ddf_a_{i}' in self._sym_approx_params:
                        mu_ddf_a_i = self._sym_approx_params[f'mu_ddf_a_{i}']
                    else:
                        mu_ddf_a_i = cs.MX.sym(f'mu_ddf_a_{i}', self.input_size, self.input_size)
                        self._sym_approx_params[f'mu_ddf_a_{i}'] = mu_ddf_a_i
                    mu_ddf_as.append(mu_ddf_a_i)

            if 'std_f_a' in self._sym_approx_params:
                std_f_a = self._sym_approx_params['std_f_a']
            else:
                std_f_a = cs.MX.sym('std_f_a', self.output_size, 1)
                self._sym_approx_params['std_f_a'] = std_f_a

            if 'std_df_a' in self._sym_approx_params:
                std_df_a = self._sym_approx_params['std_df_a']
            else:
                std_df_a = cs.MX.sym('std_df_a', self.output_size, self.input_size)
                self._sym_approx_params['std_df_a'] = std_df_a

            std_ddf_as = []
            if order == 2:
                for i in range(self.output_size):
                    if f'std_ddf_a_{i}' in self._sym_approx_params:
                        std_ddf_a_i = self._sym_approx_params[f'std_ddf_a_{i}']
                    else:
                        std_ddf_a_i = cs.MX.sym(f'std_ddf_a_{i}', self.input_size, self.input_size)
                        self._sym_approx_params[f'std_ddf_a_{i}'] = std_ddf_a_i
                    std_ddf_as.append(std_ddf_a_i)

            return [a, mu_f_a, std_f_a, mu_df_a, std_df_a] + mu_ddf_as + std_ddf_as

    def approx(self, x: cs.MX, order=1, gp_model=False):
        """
        Approximation using first or second order Taylor Expansion
        """
        approx_mx_params = self.sym_approx_params(order=order, gp_model=gp_model)
        if not gp_model:
            a = approx_mx_params[0]
            f_f_a = approx_mx_params[1]
            f_df_a = approx_mx_params[2]
            x_minus_a = x - a
            if order == 1:
                return f_f_a + cs.mtimes(f_df_a, x_minus_a)
            else:
                f_ddf_as = approx_mx_params[3:]
                second_order_term = 0.5 * cs.vcat([cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a)) for f_ddf_a in f_ddf_as])

                return (f_f_a
                        + cs.mtimes(f_df_a, x_minus_a)
                        + second_order_term)
        else:
            a = approx_mx_params[0]
            mu_f_f_a = approx_mx_params[1]
            std_f_f_a = approx_mx_params[2]
            mu_f_df_a = approx_mx_params[3]
            std_f_df_a = approx_mx_params[4]
            x_minus_a = x - a
            if order == 1:
                return cs.horzcat(
                    mu_f_f_a + cs.mtimes(mu_f_df_a, x_minus_a),
                    std_f_f_a + cs.mtimes(std_f_df_a, x_minus_a)
                ).T
            else:
                mu_f_ddf_as = approx_mx_params[5:5+self.output_size]
                std_f_ddf_as = approx_mx_params[5+self.output_size:5+2*self.output_size]
                mu_second_order_term = 0.5 * cs.vcat([cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a)) for f_ddf_a in mu_f_ddf_as])
                std_second_order_term = 0.5 * cs.vcat([cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a)) for f_ddf_a in std_f_ddf_as])

                return cs.horzcat(
                    mu_f_f_a + cs.mtimes(mu_f_df_a, x_minus_a) + mu_second_order_term,
                    std_f_f_a + cs.mtimes(std_f_df_a, x_minus_a) + std_second_order_term
                ).T

