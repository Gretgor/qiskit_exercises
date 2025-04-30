from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import QFT
import sys
from qiskit_aer import AerSimulator
import numpy as np
from math import gcd, floor, log
from fractions import Fraction
import random
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def mod_mult_gate(b,N):
    """
    Creates gate M_b for an N-state circuit
    (gate M_b, with b belonging Z*_{N}, is a gate such that
    M_b |x> = |bx> for any x in Z_{N}).

    b must be mutually prime with N, as otherwise, this will not return a
    unitary circuit. The verification of mutual primality is deferred to a
    higher level function

    The code for this function is taken straight from the IBM course material
    """
    # number of bits found according to the number of states
    n = floor(log(N-1,2)) + 1

    # matrix definition
    U = np.full((2**n,2**n),0)
    for x in range(N): 
        U[b*x % N][x] = 1
    for x in range(N,2**n): 
        U[x][x] = 1
    G = UnitaryGate(U)
    G.name = f"M_{b}"
    return G

def order_finding_circuit(a,N):
    """
    Given a and N mutually prime, with a < n, creates a quantum circuit 
    that finds the ORDER of a in Z_n. That is, an exponent r such that 
    a^r is congruent to 1 mod N.

    The code for this function is taken straight from the IBM course material
    """
    # number of bits
    n = floor(log(N-1,2)) + 1
    m = 2*n

    control = QuantumRegister(m, name = "X")
    target = QuantumRegister(n, name = "Y")
    output = ClassicalRegister(m, name = "Z")
    circuit = QuantumCircuit(control, target, output)

    # Initialize the target register to the state |1>
    circuit.x(m)

    # Add the Hadamard gates and controlled versions of the
    # multiplication gates
    for k, qubit in enumerate(control):
        circuit.h(k)
        b = pow(a,2**k,N)
        circuit.compose(
            mod_mult_gate(b,N).control(),
            qubits = [qubit] + list(target),
            inplace=True)

    # Apply the inverse QFT to the control register
    circuit.compose(
        QFT(m, inverse=True),
        qubits=control,
        inplace=True)

    # Measure the control register
    circuit.measure(control, output)

    return circuit


class IntegerFactoring:
    def __init__(self):
        self.dynamic_table = {}
        self.order_table = {}

    def find_order_classical(self, a, N):
        """
        Finds the order of a in Z_N. a and N must be mutually prime
        """
        cur_exp = 1
        while pow(a,cur_exp,N) != 1:
            cur_exp += 1
        self.order_table[(a,N)] = cur_exp
        return cur_exp

    def find_order(self, a, N, classical=False, real_cloud=True):
        """
        Given a, N, mutually prime and such that a < N, applies the quantum
        order finding circuit to obtain the order of a in Z_N.
        
        Set flag "classical" for classical order finding
        """
        if (a,N) in self.order_table.keys():
            return self.order_table[(a,N)]
        if classical:
            print(f"finding order for {a},{N} in classical way", file=sys.stderr)
            return self.find_order_classical(a, N)
        n = floor(log(N-1,2)) + 1
        m = 2*n
        circuit = order_finding_circuit(a,N)

        # loops indefinitely until it finds the order.
        # does not take as long as it seems, since the probability of finding
        # the correct order is very high.
        transpiled_circuit = transpile(circuit,AerSimulator())
        while not real_cloud:
            result = AerSimulator().run(
                transpiled_circuit,
                shots=1,
                memory=True).result()
            y = int(result.get_memory()[0],2)
            r = Fraction(y/2**m).limit_denominator(N).denominator
            if pow(a,r,N)==1: 
                self.order_table[(a,N)] = r
                return r

        # if real_cloud is set to true, use the IBM cloud for actual quantum computing
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        circuit_to_run = pm.run(circuit)

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 1024
        job = sampler.run([circuit_to_run])
        my_result = job.result()[0].data.Z.array
        for r in my_result:
            if pow(a,int(r[0]),N)==1: 
                self.order_table[(a,N)] = int(r[0])
                return int(r[0])


    def factor_integer(self, N, classical=False):
        """ 
        Breaks an integer into prime factors, and returns the factors as a list
    
        args: N: integer to factor
        returns: list of prime factors
        """
        # with a failure chance of less than 1/2, we can safely assume, with less than 0.1% probability of mistake,
        # that any number for which we fail to find a non-trivial factor after trying 10 times is, itself, prime
        MAX_ATTEMPTS = 10
    
        if N == 1:
            return []
        # if the factoring for N has already been computed, just use it
        if N in self.dynamic_table.keys():
            return self.dynamic_table[N]
        if N % 2 == 0:
            # N is even, extract the 2 and make a recursive call
            self.dynamic_table[N] = [2] + self.factor_integer(N//2, classical)
            return self.dynamic_table[N]        
    
        # check if the number is a power (if so, iterate over the power)
        for k in range(2,round(log(N,2))+1):
            d = int(round(N ** (1/k)))
            if d**k == N:
                # Number is a power, iterate on the root
                self.dynamic_table[N] = self.factor_integer(d, classical) + self.factor_integer(N//d, classical)
                return self.dynamic_table[N]
        factor = -1
        num_its = 0
        while factor == -1 and num_its < MAX_ATTEMPTS:
            a = random.randint(2,N-1)
            d = gcd(a,N)
            if d > 1:
                factor = d
            else:
                r = self.find_order(a, N, classical)
                if r % 2 == 0:
                    x = pow(a,r//2,N) - 1
                    d = gcd(x,N)
                    if d > 1: 
                        factor = d
            num_its += 1
        if factor > -1:
            self.dynamic_table[N] = self.factor_integer(d, classical) + self.factor_integer(N//d, classical)
            return self.dynamic_table[N]
        else:
            # if no factor was ever found, then we can assume N itself is prime
            self.dynamic_table[N] = [N]
            return [N]

def show_factoring(N, classical=False):
    """
    factors integer N and shows the prime decomposition in a human-readable form

    Args: N: the number to factor
          classical: set for classical order finding (i.e. not via quantum circuit)
    Returns: a string containing a human-readable form of the prime decomposition
    """
    factorer = IntegerFactoring()
    factor_list = factorer.factor_integer(N, classical)
    factor_list.sort()
    cur_factor = -1
    cur_exponent = 0
    final_string = ""
    for element in factor_list:
        if cur_factor != element:
            if cur_factor != -1:
                if len(final_string) > 0:
                    final_string += f"*{cur_factor}^{cur_exponent}"
                else:
                    final_string += f"{cur_factor}^{cur_exponent}"
            cur_exponent = 1
            cur_factor = element
        else:
            cur_exponent += 1
    if len(final_string) > 0:
        final_string += f"*{cur_factor}^{cur_exponent}"
    else:
        final_string += f"{cur_factor}^{cur_exponent}"
    return final_string

if __name__ == '__main__':
    number_to_factor = int(sys.argv[1])
    classical = len(sys.argv) > 2 and sys.argv[2] == '1'
    print(f"classical: {classical}", file=sys.stderr)
    print(f"Prime decomposition of number {number_to_factor}:")
    print(show_factoring(number_to_factor, classical))
    
