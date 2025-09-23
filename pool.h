// thread_pool.hpp
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool
{
public:
    explicit ThreadPool(std::size_t thread_count = std::thread::hardware_concurrency())
        : stop_flag_(false), active_tasks_(0)
    {

        if (thread_count == 0)
        {
            thread_count = 1;
        }

        workers_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i)
        {
            workers_.emplace_back(&ThreadPool::worker_loop, this);
        }
    }

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    ~ThreadPool()
    {
        shutdown();
    }

    // Enqueue a task and return a future for the result
    template <typename F, typename... Args>
    auto enqueue(F &&func, Args &&...args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    {

        using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

        // Create a packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();

        // Add task to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);

            if (stop_flag_)
            {
                throw std::runtime_error("Cannot enqueue task on stopped ThreadPool");
            }

            task_queue_.emplace([task]
                                { (*task)(); });
        }

        condition_.notify_one();
        return result;
    }

    // Wait for all tasks to complete
    void wait_idle()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        idle_condition_.wait(lock, [this]
                             { return task_queue_.empty() && active_tasks_ == 0; });
    }

    // Get number of worker threads
    std::size_t thread_count() const noexcept
    {
        return workers_.size();
    }

    // Check if the pool is stopped
    bool is_stopped() const noexcept
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return stop_flag_;
    }

    // Get number of queued tasks
    std::size_t queue_size() const noexcept
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return task_queue_.size();
    }

private:
    using Task = std::function<void()>;

    void worker_loop()
    {
        while (true)
        {
            Task task;

            // Get next task
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait for work or stop signal
                condition_.wait(lock, [this]
                                { return stop_flag_ || !task_queue_.empty(); });

                // Exit if stopping and no more work
                if (stop_flag_ && task_queue_.empty())
                {
                    return;
                }

                // Get task and mark as active
                task = std::move(task_queue_.front());
                task_queue_.pop();
                ++active_tasks_;
            }

            // Execute task (outside of lock)
            try
            {
                task();
            }
            catch (...)
            {
                // Silently ignore exceptions - consider logging in production
            }

            // Mark task as completed
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                --active_tasks_;

                // Notify if all work is done
                if (active_tasks_ == 0 && task_queue_.empty())
                {
                    idle_condition_.notify_all();
                }
            }
        }
    }

    void shutdown()
    {
        // Signal all workers to stop
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_flag_ = true;
        }

        condition_.notify_all();

        // Wait for all workers to finish
        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    // Thread synchronization
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable idle_condition_;

    // Worker threads and task queue
    std::vector<std::thread> workers_;
    std::queue<Task> task_queue_;

    // State variables
    bool stop_flag_;
    std::size_t active_tasks_;
};