"""
Meta-learning demonstration showing GAIA's ability to learn how to learn.

This example demonstrates GAIA's meta-learning capabilities where the system
learns optimal plasticity parameters across multiple tasks.
"""

import numpy as np
from gaia.layers.hebbian import HebbianCore
from gaia.layers.temporal import TemporalLayer
from gaia.plasticity.controller import PlasticityController
from gaia.meta_learning.optimizer import MetaOptimizer
from gaia.meta_learning.metrics import evaluate_meta_learning_curve
from gaia.utils.logging import setup_logging
from gaia.utils.visualization import plot_learning_curve, plot_plasticity_parameters

def meta_learning_demo():
    """Demonstration of GAIA's meta-learning capabilities."""
    print("🧠 GAIA Meta-Learning Demo")
    print("=" * 50)

    # Setup logging
    logger = setup_logging('gaia_meta', 'INFO')
    logger.info("Starting GAIA meta-learning demo")

    # Create target modules
    logger.info("Creating target modules...")
    hebbian_core = HebbianCore(input_size=20, output_size=40, plasticity_rule='hebbian')
    temporal_layer = TemporalLayer(input_size=40, output_size=80, time_window=5)

    # Create plasticity controller
    logger.info("Creating PlasticityController...")
    plasticity_controller = PlasticityController(
        target_modules=[hebbian_core, temporal_layer],
        adaptation_rate=0.01,
        exploration_noise=0.1
    )

    # Create meta-optimizer
    logger.info("Creating MetaOptimizer...")
    meta_optimizer = MetaOptimizer(plasticity_controller)

    print(f"PlasticityController: {plasticity_controller}")
    print(f"MetaOptimizer: {meta_optimizer}")

    # Define task distribution
    def create_task(task_id: int):
        """Create a simple task function with different characteristics."""
        def task(step: int) -> float:
            """
            Task function that returns performance based on step and task characteristics.

            Args:
                step: Current step in adaptation

            Returns:
                Performance metric
            """
            # Base performance that depends on task difficulty
            base_performance = 0.4 + 0.1 * task_id

            # Performance improves with adaptation steps
            improvement = 0.02 * step

            # Add task-specific noise
            noise = 0.05 * (task_id + 1) * np.random.randn()

            # Final performance
            performance = base_performance + improvement + noise

            return np.clip(performance, 0.0, 1.0)

        return task

    # Create multiple tasks with different characteristics
    num_tasks = 5
    tasks = [create_task(i) for i in range(num_tasks)]

    print(f"\nCreated {num_tasks} tasks with different characteristics")

    # Run meta-learning
    num_episodes = 30
    logger.info(f"Running {num_episodes} meta-learning episodes...")
    print(f"\nRunning {num_episodes} meta-learning episodes...")

    meta_optimizer.meta_train(num_episodes, tasks)

    # Evaluate meta-learning performance
    metrics = meta_optimizer.evaluate_meta_performance()
    print(f"\n📊 Meta-Learning Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Visualize results
    print("\n📊 Visualizing Results...")

    # Plot meta-learning curve
    learning_history = meta_optimizer.learning_history
    if learning_history:
        plot_learning_curve(learning_history, title="Meta-Learning Performance")

        # Evaluate learning curve
        curve_metrics = evaluate_meta_learning_curve(learning_history)
        print(f"\n📈 Learning Curve Analysis:")
        for metric, value in curve_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Plot parameter evolution
    param_history = plasticity_controller.get_param_history()
    if param_history:
        param_names = ['param_' + str(i) for i in range(len(param_history[0]))]
        param_dict_history = [{param_names[j]: param_history[i][j] for j in range(len(param_history[i]))}
                             for i in range(len(param_history))]

        plot_plasticity_parameters(param_dict_history, title="Meta-Learning Parameter Evolution")

    # Display final configuration
    meta_config = meta_optimizer.get_config()
    print(f"\n📝 Final Meta-Optimizer Configuration:")
    for key, value in meta_config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ Meta-learning demo completed successfully!")
    logger.info("GAIA meta-learning demo completed successfully")

def task_adaptation_demo():
    """Demonstrate adaptation to different tasks."""
    print("\n🎯 Task Adaptation Demo")
    print("=" * 50)

    # Create a HebbianCore that will adapt to different tasks
    core = HebbianCore(input_size=10, output_size=20, plasticity_rule='hebbian')

    # Define different task patterns
    tasks = {
        'pattern1': np.random.randn(100, 10),  # Random pattern
        'pattern2': np.ones((100, 10)) * 0.5,  # Constant pattern
        'pattern3': np.eye(10)[np.random.randint(0, 10, 100)]  # One-hot pattern
    }

    results = {}

    for task_name, task_data in tasks.items():
        print(f"\nAdapting to {task_name}...")

        # Reset core for new task
        core.reset_state()

        # Adapt to task
        for t in range(100):
            output = core.forward(task_data[t])
            core.update()

        # Store final weights
        results[task_name] = core.get_weights().copy()

        # Calculate adaptation metrics
        weight_changes = np.linalg.norm(results[task_name] - core.weights)
        print(f"{task_name}: weight changes = {weight_changes:.4f}")

    # Compare final weights across tasks
    print("\n📊 Task Adaptation Results:")
    for task_name, weights in results.items():
        print(f"{task_name}: weight norm = {np.linalg.norm(weights):.4f}")

    # Calculate differences between task adaptations
    task_names = list(results.keys())
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            diff = np.linalg.norm(results[task_names[i]] - results[task_names[j]])
            print(f"Difference between {task_names[i]} and {task_names[j]}: {diff:.4f}")

def multi_task_learning():
    """Demonstrate learning across multiple tasks simultaneously."""
    print("\n🤹 Multi-Task Learning Demo")
    print("=" * 50)

    # Create multiple HebbianCores for different tasks
    cores = {
        'task1': HebbianCore(input_size=10, output_size=20, plasticity_rule='hebbian'),
        'task2': HebbianCore(input_size=10, output_size=20, plasticity_rule='oja'),
        'task3': HebbianCore(input_size=10, output_size=20, plasticity_rule='bcm')
    }

    # Generate task data
    task_data = {
        'task1': np.random.randn(100, 10),
        'task2': np.random.randn(100, 10) * 0.5,
        'task3': np.random.randn(100, 10) * 2.0
    }

    # Train each core on its task
    for task_name, core in cores.items():
        print(f"\nTraining {task_name}...")

        for t in range(100):
            output = core.forward(task_data[task_name][t])
            core.update()

        # Store final performance
        final_weights = core.get_weights()
        print(f"{task_name}: final weight norm = {np.linalg.norm(final_weights):.4f}")

    # Compare learned representations
    print("\n📊 Multi-Task Learning Results:")
    for task_name, core in cores.items():
        weights = core.get_weights()
        print(f"{task_name} ({core.plasticity_rule}): norm = {np.linalg.norm(weights):.4f}")

if __name__ == "__main__":
    meta_learning_demo()
    task_adaptation_demo()
    multi_task_learning()