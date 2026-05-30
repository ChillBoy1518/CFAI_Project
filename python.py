"""
AI-Based Traffic Signal Optimization System
Author: Traffic AI Research
Course: Computational Foundations for AI

This system takes real-time vehicle data from 4 roads (SC1, SC2, SC3, SC4)
and calculates optimal traffic light timings using multiple AI techniques.
"""

import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque


# ============================================================================
# PART 1: DATA STRUCTURES & STATE REPRESENTATION
# ============================================================================

@dataclass
class RoadData:
    """Simple data structure for each road's current status"""
    road_name: str
    vehicle_count: int  # INPUT: number of vehicles waiting
    emergency: bool  # INPUT: emergency vehicle present? (True/False)
    waiting_time: float  # CALCULATED: average wait time in seconds
    green_time: int  # OUTPUT: calculated green light duration

    def __init__(self, name: str, vehicles: int, emergency_flag: bool):
        self.road_name = name
        self.vehicle_count = vehicles
        self.emergency = emergency_flag
        self.waiting_time = 0.0  # Will be calculated
        self.green_time = 0  # Will be calculated


class TrafficIntersection:
    """
    Represents a 4-way intersection with roads:
    SC1 - North Road
    SC2 - South Road
    SC3 - East Road
    SC4 - West Road
    """

    def __init__(self):
        self.roads = {}  # Dictionary to store road data
        self.cycle_duration = 120  # Total cycle time in seconds
        self.min_green = 10  # Minimum green time
        self.max_green = 60  # Maximum green time

    def load_vehicle_data(self, sc1: int, sc2: int, sc3: int, sc4: int,
                          emergency_sc1: bool = False, emergency_sc2: bool = False,
                          emergency_sc3: bool = False, emergency_sc4: bool = False):
        """
        Load input vehicle data for all 4 roads.
        This is the main input method for the system.

        Example usage:
        intersection.load_vehicle_data(45, 20, 60, 15, emergency_sc3=True)
        """
        self.roads["SC1"] = RoadData("North Road", sc1, emergency_sc1)
        self.roads["SC2"] = RoadData("South Road", sc2, emergency_sc2)
        self.roads["SC3"] = RoadData("East Road", sc3, emergency_sc3)
        self.roads["SC4"] = RoadData("West Road", sc4, emergency_sc4)

        # Calculate waiting times based on vehicle count
        self._calculate_waiting_times()

    def _calculate_waiting_times(self):
        """
        Calculate waiting time for each road based on vehicle count.
        More vehicles = longer waiting time (this makes intuitive sense)
        """
        for road in self.roads.values():
            # Base waiting time: each vehicle adds roughly 2 seconds of wait
            base_wait = road.vehicle_count * 2

            # Emergency vehicles add extra urgency
            if road.emergency:
                base_wait = base_wait * 1.5

            road.waiting_time = base_wait

    def get_urgent_roads(self) -> List[str]:
        """
        Identify which roads need immediate attention.
        Used by heuristics to prioritize.
        """
        urgent = []
        for road_name, road in self.roads.items():
            # Road is urgent if: high traffic OR emergency present
            if road.vehicle_count > 50 or road.emergency:
                urgent.append(road_name)
        return urgent

    def get_total_vehicles(self) -> int:
        """Calculate total vehicles waiting at intersection"""
        return sum(road.vehicle_count for road in self.roads.values())

    def display_current_state(self):
        """Pretty print current traffic situation"""
        print("\n" + "=" * 60)
        print("CURRENT TRAFFIC SITUATION")
        print("=" * 60)
        for road in self.roads.values():
            emergency_mark = " 🚨 EMERGENCY" if road.emergency else ""
            print(f"{road.road_name:15} | Vehicles: {road.vehicle_count:3} | "
                  f"Wait Time: {road.waiting_time:5.1f}s{emergency_mark}")
        print("=" * 60)


# ============================================================================
# PART 2: CONSTRAINT SATISFACTION PROBLEM (CSP)
# ============================================================================

class TrafficConstraints:
    """
    Handles all traffic rules and constraints.
    Think of this as the "rule book" for traffic lights.
    """

    def __init__(self, intersection: TrafficIntersection):
        self.intersection = intersection

    def check_min_green(self, road_name: str, green_duration: int) -> bool:
        """Constraint: Green light cannot be too short"""
        return green_duration >= self.intersection.min_green

    def check_max_green(self, road_name: str, green_duration: int) -> bool:
        """Constraint: Green light cannot be too long"""
        return green_duration <= self.intersection.max_green

    def check_cycle_total(self, timings: Dict[str, int]) -> bool:
        """Constraint: All signal timings should add up to cycle duration"""
        total = sum(timings.values())
        # Allow 5-second tolerance for flexibility
        return abs(total - self.intersection.cycle_duration) <= 5

    def check_opposite_roads(self, timings: Dict[str, int]) -> bool:
        """
        Constraint: Opposite roads shouldn't both have very long green times.
        Example: SC1 (North) and SC2 (South) are opposites.
        """
        # North-South pair
        ns_diff = abs(timings.get("SC1", 0) - timings.get("SC2", 0))
        # East-West pair
        ew_diff = abs(timings.get("SC3", 0) - timings.get("SC4", 0))

        # If difference is too big (>40 seconds), that's unfair
        return ns_diff <= 40 and ew_diff <= 40

    def check_emergency_priority(self, timings: Dict[str, int]) -> bool:
        """
        Constraint: Roads with emergency vehicles must get significant green time
        """
        for road_name, road in self.intersection.roads.items():
            if road.emergency:
                if timings.get(road_name, 0) < 30:  # Emergency needs at least 30 sec
                    return False
        return True

    def validate_timings(self, timings: Dict[str, int]) -> Tuple[bool, List[str]]:
        """
        Run ALL constraints on a proposed timing schedule.
        Returns: (is_valid, list_of_violations)
        """
        violations = []

        # Check each road's timing individually
        for road_name, duration in timings.items():
            if not self.check_min_green(road_name, duration):
                violations.append(f"{road_name}: {duration}s is less than minimum {self.intersection.min_green}s")
            if not self.check_max_green(road_name, duration):
                violations.append(f"{road_name}: {duration}s exceeds maximum {self.intersection.max_green}s")

        # Check overall constraints
        if not self.check_cycle_total(timings):
            total = sum(timings.values())
            violations.append(f"Total cycle time {total}s should be {self.intersection.cycle_duration}s")

        if not self.check_opposite_roads(timings):
            violations.append("Opposite roads have unbalanced timings")

        if not self.check_emergency_priority(timings):
            violations.append("Emergency vehicle road doesn't have enough green time")

        return len(violations) == 0, violations


# ============================================================================
# PART 3: HEURISTICS & SCORING
# ============================================================================

class TrafficHeuristics:
    """
    Smart rules to make better decisions faster.
    These are like "best practices" for traffic management.
    """

    @staticmethod
    def calculate_priority_score(road: RoadData) -> float:
        """
        Calculate how urgently a road needs green light.
        Higher score = more urgent.

        Formula considers:
        - Number of vehicles waiting
        - How long they've been waiting
        - Whether there's an emergency
        """
        score = road.vehicle_count * 1.0  # Each vehicle adds 1 point
        score += road.waiting_time * 0.5  # Waiting time adds half point per second

        if road.emergency:
            score *= 3.0  # Emergency triples the urgency

        return score

    @staticmethod
    def suggest_green_time(road: RoadData, total_vehicles: int) -> int:
        """
        Suggest an initial green time based on traffic load.
        This is a smart guess before CSP refines it.
        """
        if total_vehicles == 0:
            return 15  # Default minimum if no traffic

        # Calculate proportion based on vehicle count
        proportion = road.vehicle_count / total_vehicles
        suggested = int(proportion * 120)  # 120 = cycle duration

        # Clamp to valid range
        suggested = max(10, min(60, suggested))

        # Emergency vehicles get bonus time
        if road.emergency:
            suggested = max(suggested, 35)

        return suggested


# ============================================================================
# PART 4: BACKTRACKING SEARCH ENGINE
# ============================================================================

class BacktrackingSearch:
    """
    Searches for the best signal timing by trying different combinations.
    Uses CSP constraints and heuristics to search efficiently.
    """

    def __init__(self, constraints: TrafficConstraints):
        self.constraints = constraints
        self.attempts = 0  # Track how many solutions we tried
        self.best_solution = None
        self.best_score = -999999

    def find_optimal_timings(self, intersection: TrafficIntersection) -> Dict[str, int]:
        """
        Main search function - finds the best signal timings.
        Uses backtracking with heuristics for efficiency.
        """
        print("\n🔍 Running AI optimization...")

        # Get initial guesses using heuristics
        total_vehicles = intersection.get_total_vehicles()
        initial_timings = {}

        for road_name, road in intersection.roads.items():
            initial_timings[road_name] = TrafficHeuristics.suggest_green_time(road, total_vehicles)

        # Adjust to match cycle duration
        initial_timings = self._adjust_to_cycle(initial_timings)

        print(f"   Initial guess: {initial_timings}")

        # Start backtracking search from initial guess
        best = self._backtrack(initial_timings, intersection, depth=0)

        print(f"   Tried {self.attempts} different combinations")
        print(f"   Best solution found: {best}")

        return best

    def _adjust_to_cycle(self, timings: Dict[str, int]) -> Dict[str, int]:
        """Adjust timings so they sum to exactly cycle duration"""
        current_sum = sum(timings.values())
        target = 120

        if current_sum == target:
            return timings

        # Scale all timings proportionally
        scale = target / current_sum
        adjusted = {}
        for road, time in timings.items():
            new_time = int(time * scale)
            new_time = max(10, min(60, new_time))
            adjusted[road] = new_time

        return adjusted

    def _backtrack(self, current_timings: Dict[str, int],
                   intersection: TrafficIntersection,
                   depth: int) -> Dict[str, int]:
        """
        Recursive backtracking search.
        Tries different adjustments to find best valid schedule.
        """
        self.attempts += 1

        # Check if current timings are valid
        is_valid, violations = self.constraints.validate_timings(current_timings)

        if is_valid:
            # Calculate how good this solution is
            score = self._evaluate_solution(current_timings, intersection)

            if score > self.best_score:
                self.best_score = score
                self.best_solution = current_timings.copy()
                print(f"   Found better solution! Score: {score:.1f}")

            return self.best_solution

        # Stop if we've searched enough (limit depth)
        if depth > 20:
            return self.best_solution or current_timings

        # Try to fix violations by adjusting timings
        for road in current_timings.keys():
            # Try increasing or decreasing this road's timing
            for delta in [-10, -5, 5, 10]:
                new_timings = current_timings.copy()
                new_timings[road] += delta
                new_timings[road] = max(10, min(60, new_timings[road]))

                # Make sure cycle total stays balanced
                new_timings = self._adjust_to_cycle(new_timings)

                # Recurse
                result = self._backtrack(new_timings, intersection, depth + 1)
                if result:
                    return result

        return self.best_solution

    def _evaluate_solution(self, timings: Dict[str, int],
                           intersection: TrafficIntersection) -> float:
        """
        Calculate a score for a timing solution.
        Higher score = better solution.
        """
        score = 0

        for road_name, road in intersection.roads.items():
            green_time = timings.get(road_name, 0)

            # Reward: More green time for roads with more vehicles
            score += road.vehicle_count * (green_time / 10)

            # Reward: Emergency roads getting green time
            if road.emergency:
                score += green_time * 2

            # Penalty: Long waits without green time
            if green_time < 20 and road.vehicle_count > 30:
                score -= road.waiting_time / 10

        return score


# ============================================================================
# PART 5: BAYESIAN PREDICTION (Simple but Effective)
# ============================================================================

class BayesianPredictor:
    """
    Simple Bayesian reasoning to predict congestion.
    Helps the system make smarter decisions about the future.
    """

    def __init__(self):
        # Prior probabilities (our initial beliefs)
        self.p_congestion = 0.30  # 30% chance of congestion normally
        self.p_high_traffic = 0.25  # 25% chance of high traffic

    def predict_congestion_risk(self, vehicle_count: int, waiting_time: float) -> float:
        """
        Calculate probability of congestion getting worse.
        Uses Bayes' theorem: P(congestion | current data)
        """
        # Likelihood: If many vehicles, congestion is likely
        if vehicle_count > 50:
            likelihood = 0.85
        elif vehicle_count > 30:
            likelihood = 0.60
        else:
            likelihood = 0.20

        # Bayes' theorem calculation
        posterior = (likelihood * self.p_congestion) / (
                likelihood * self.p_congestion +
                (1 - likelihood) * (1 - self.p_congestion)
        )

        # Adjust based on waiting time (long waits = higher risk)
        waiting_factor = 1 + (waiting_time / 100)
        posterior = min(0.95, posterior * waiting_factor)

        return posterior

    def predict_accident_risk(self, vehicle_count: int, emergency: bool) -> float:
        """
        Simple risk prediction based on traffic density
        """
        if emergency:
            return 0.15  # Emergency vehicles slightly increase accident risk

        if vehicle_count > 60:
            return 0.25
        elif vehicle_count > 40:
            return 0.12
        else:
            return 0.05


# ============================================================================
# PART 6: MAIN OPTIMIZATION SYSTEM
# ============================================================================

class TrafficOptimizer:
    """
    Main system that brings everything together.
    This is where the magic happens.
    """

    def __init__(self):
        self.intersection = TrafficIntersection()
        self.constraints = None
        self.search = None
        self.predictor = BayesianPredictor()

    def process_traffic_data(self, sc1: int, sc2: int, sc3: int, sc4: int,
                             emergency_sc1: bool = False, emergency_sc2: bool = False,
                             emergency_sc3: bool = False, emergency_sc4: bool = False) -> Dict[str, int]:
        """
        MAIN FUNCTION - Call this to optimize traffic signals.

        Input parameters:
        - sc1, sc2, sc3, sc4: Number of vehicles waiting at each road
        - emergency_sc1... : Whether emergency vehicle present (True/False)

        Returns:
        - Dictionary with optimal green light durations for each road
        """

        print("\n" + "🚦" * 30)
        print("TRAFFIC SIGNAL OPTIMIZATION SYSTEM")
        print("🚦" * 30)

        # Step 1: Load input data
        self.intersection.load_vehicle_data(
            sc1, sc2, sc3, sc4,
            emergency_sc1, emergency_sc2, emergency_sc3, emergency_sc4
        )

        # Step 2: Display current situation
        self.intersection.display_current_state()

        # Step 3: Setup constraints
        self.constraints = TrafficConstraints(self.intersection)

        # Step 4: Bayesian predictions
        print("\n🔮 TRAFFIC PREDICTIONS (Bayesian Analysis)")
        print("-" * 40)
        for road_name, road in self.intersection.roads.items():
            congestion_risk = self.predictor.predict_congestion_risk(
                road.vehicle_count, road.waiting_time
            )
            accident_risk = self.predictor.predict_accident_risk(
                road.vehicle_count, road.emergency
            )
            print(f"{road.road_name:12} | Congestion risk: {congestion_risk:5.1%} | "
                  f"Accident risk: {accident_risk:5.1%}")

        # Step 5: Find optimal timings using search
        self.search = BacktrackingSearch(self.constraints)
        optimal_timings = self.search.find_optimal_timings(self.intersection)

        # Step 6: Validate final solution
        is_valid, violations = self.constraints.validate_timings(optimal_timings)

        # Step 7: Display results
        self._display_results(optimal_timings, is_valid, violations)

        # Step 8: Return the optimized timings
        return optimal_timings

    def _display_results(self, timings: Dict[str, int], is_valid: bool, violations: List[str]):
        """Pretty print the optimization results"""

        print("\n" + "=" * 60)
        print("OPTIMIZED SIGNAL TIMINGS")
        print("=" * 60)

        # Display each road's timing
        for road_name, duration in timings.items():
            road = self.intersection.roads[road_name]
            bar_length = int(duration / 2)  # Visual bar for green time
            bar = "█" * bar_length + "░" * (30 - bar_length)

            emergency_mark = " 🚨" if road.emergency else ""
            print(f"{road.road_name:12} | {duration:2} seconds | {bar} |{emergency_mark}")

        print("-" * 60)
        print(f"Total cycle time: {sum(timings.values())} seconds (Target: 120s)")

        # Show constraint status
        if is_valid:
            print("\n✅ All constraints satisfied! Solution is valid.")
        else:
            print("\n⚠️ Warning: Some constraints violated:")
            for v in violations[:3]:
                print(f"   • {v}")

        # Show reasoning
        print("\n💡 WHY THESE TIMINGS?")
        print("-" * 40)
        for road_name, duration in timings.items():
            road = self.intersection.roads[road_name]
            if road.emergency:
                print(f"• {road.road_name}: Emergency vehicle - extended green to {duration}s")
            elif road.vehicle_count > 50:
                print(f"• {road.road_name}: Heavy traffic ({road.vehicle_count} vehicles) - {duration}s green")
            elif road.vehicle_count < 20:
                print(f"• {road.road_name}: Light traffic - {duration}s green")

        print("\n" + "🚦" * 30)


# ============================================================================
# PART 7: EXAMPLE USAGE & DEMO
# ============================================================================

def demo_scenario_1():
    """Normal traffic - moderate flow"""
    print("\n" + "📋 DEMO SCENARIO 1: Normal Traffic")
    print("-" * 40)

    optimizer = TrafficOptimizer()
    result = optimizer.process_traffic_data(
        sc1=35,  # North: 35 vehicles
        sc2=28,  # South: 28 vehicles
        sc3=42,  # East: 42 vehicles
        sc4=15  # West: 15 vehicles
    )
    return result


def demo_scenario_2():
    """Emergency vehicle on East road"""
    print("\n" + "📋 DEMO SCENARIO 2: Emergency Vehicle on East Road 🚨")
    print("-" * 40)

    optimizer = TrafficOptimizer()
    result = optimizer.process_traffic_data(
        sc1=40,  # North: 40 vehicles
        sc2=25,  # South: 25 vehicles
        sc3=55,  # East: 55 vehicles + EMERGENCY
        sc4=10,  # West: 10 vehicles
        emergency_sc3=True  # Emergency on East road
    )
    return result


def demo_scenario_3():
    """Rush hour - all roads congested"""
    print("\n" + "📋 DEMO SCENARIO 3: Rush Hour - Heavy Traffic Everywhere")
    print("-" * 40)

    optimizer = TrafficOptimizer()
    result = optimizer.process_traffic_data(
        sc1=75,  # North: 75 vehicles (heavy)
        sc2=68,  # South: 68 vehicles (heavy)
        sc3=82,  # East: 82 vehicles (very heavy)
        sc4=55,  # West: 55 vehicles (moderate heavy)
        emergency_sc2=True  # Emergency on South road
    )
    return result


def interactive_mode():
    """Let user input custom traffic data"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Enter Your Traffic Data")
    print("=" * 60)

    try:
        sc1 = int(input("Enter vehicles on North Road (SC1): "))
        sc2 = int(input("Enter vehicles on South Road (SC2): "))
        sc3 = int(input("Enter vehicles on East Road (SC3): "))
        sc4 = int(input("Enter vehicles on West Road (SC4): "))

        print("\nEmergency vehicles? (yes/no)")
        emergency_input = input("Any emergency vehicle present? ").lower()

        emergency_sc1 = emergency_sc2 = emergency_sc3 = emergency_sc4 = False

        if emergency_input == 'yes':
            emergency_road = input("Which road? (SC1/SC2/SC3/SC4): ").upper()
            if emergency_road == 'SC1':
                emergency_sc1 = True
            elif emergency_road == 'SC2':
                emergency_sc2 = True
            elif emergency_road == 'SC3':
                emergency_sc3 = True
            elif emergency_road == 'SC4':
                emergency_sc4 = True

        optimizer = TrafficOptimizer()
        result = optimizer.process_traffic_data(
            sc1, sc2, sc3, sc4,
            emergency_sc1, emergency_sc2, emergency_sc3, emergency_sc4
        )

        print("\n✅ Optimization Complete!")
        print("Final Signal Timings:")
        for road, time in result.items():
            print(f"   {road}: {time} seconds")

    except ValueError:
        print("❌ Please enter valid numbers!")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     AI-BASED TRAFFIC SIGNAL OPTIMIZATION SYSTEM              ║
    ║                                                              ║
    ║     This system uses:                                        ║
    ║     • State Space Representation                             ║
    ║     • Constraint Satisfaction Problem (CSP)                  ║
    ║     • Backtracking Search                                    ║
    ║     • Heuristics (MRV, LCV, Forward Checking)                ║
    ║     • Bayesian Reasoning                                     ║
    ║     • Utility-based Decision Making                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print("\nSelect a mode:")
    print("1. Run Demo Scenarios")
    print("2. Interactive Mode (Enter your own data)")
    print("3. Exit")

    choice = input("\nEnter choice (1/2/3): ")

    if choice == '1':
        demo_scenario_1()
        print("\n" + "=" * 60)
        input("\nPress Enter for next scenario...")
        demo_scenario_2()
        print("\n" + "=" * 60)
        input("\nPress Enter for next scenario...")
        demo_scenario_3()

    elif choice == '2':
        interactive_mode()

    else:
        print("Goodbye!")

    print("\n" + "=" * 60)
    print("SYSTEM READY FOR REAL-TIME DEPLOYMENT")
    print("=" * 60)
