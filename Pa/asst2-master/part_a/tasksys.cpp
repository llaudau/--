#include "tasksys.h"
#include "itasksys.h"

#include <ostream>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>  
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <string>
using namespace std;

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */


const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
        // cout<<num_total_tasks<<i<<endl;
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    this->num_threads=num_threads;
    
    
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::thread* threads=nullptr;
    
    int num_join;
    if (num_threads>num_total_tasks){
        threads=new std::thread[num_total_tasks];
        num_join=num_total_tasks;
        for(int i=0;i<num_total_tasks;i++){
            threads[i]=std::thread([=](){runnable->runTask(i,num_total_tasks);});
        }
    }
    else{
        vector<int> tasklist(num_total_tasks);
        for(int i=0; i<num_total_tasks; i++){
            tasklist[i] = i;
        }

        shuffle(tasklist.begin(), tasklist.end(), default_random_engine());
        threads=new std::thread[num_threads];
        int task_p_thread=num_total_tasks/num_threads;
        num_join=num_threads;
        for (int i = 0; i < num_threads; i++){
            int start_index = i * task_p_thread;
            int end_index;
            if(i == num_threads-1)
                { end_index = num_total_tasks;}
            else{
                end_index=(i+1)*task_p_thread;
            }
            threads[i]=std::thread([=](){for(int j =start_index;j<end_index;j++){
                runnable->runTask(tasklist[j],num_total_tasks);
            }
        });
            
        }
    }
    for(int i =0; i<num_join;i++){
        threads[i].join();
    }
    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }

}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

 TasksState::TasksState() {
    mutex_ = new std::mutex();
    finished_ = new std::condition_variable();
    finishedMutex_ = new std::mutex();
    runnable_ = nullptr;
    finished_tasks_ = -1;
    left_tasks_ = -1;
    num_total_tasks_ = -1;
}

TasksState::~TasksState() {
    delete mutex_;
    delete finished_;
    delete finishedMutex_;
}
const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}


void TaskSystemParallelThreadPoolSpinning::Threads_cant_sleep(){
    int id;
    int total;
    while (true){
        if (stop) break;
        state_->mutex_->lock();
        total=state_->num_total_tasks_;
        id=total-state_->left_tasks_;
        if (id<total)state_->left_tasks_--;
        state_->mutex_->unlock();
        if (id<total){
            state_->runnable_->runTask(id, total);
            state_->mutex_->lock();
            state_->finished_tasks_++;
            if (state_->finished_tasks_==total){
                state_->mutex_->unlock();
                state_->finishedMutex_->lock();
                state_->finishedMutex_->unlock();
                state_->finished_->notify_all();

            }
            else {
            state_->mutex_->unlock();
            }
            
        }
        
    }

};
TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads)
    : ITaskSystem(num_threads){

    state_=new TasksState;
    stop=false;
    num_threads_=num_threads;
    threads_pool_=new std::thread[num_threads];
    for (int i=0;i<num_threads;i++){
        threads_pool_[i]=std::thread(&TaskSystemParallelThreadPoolSpinning::Threads_cant_sleep,this);
    }
    }
    



TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    stop=true;
    for(int i=0;i<num_threads_;i++){
        threads_pool_[i].join();
    }
    delete[] threads_pool_;
    delete state_;

}



void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    
    std::unique_lock<std::mutex> lk(*(state_->finishedMutex_));
    state_->mutex_->lock();
    state_->finished_tasks_ = 0;
    state_->left_tasks_ = num_total_tasks;
    state_->num_total_tasks_ = num_total_tasks;
    state_->runnable_ = runnable;
    state_->mutex_->unlock();
    // go to sleep, wait for all the tasks to finish
    //std::cerr << "go to sleep" << std::endl;
    state_->finished_->wait(lk); 

}



TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    
    state_=new TasksState;
    stop=false;
    num_threads_=num_threads;
    threads_pool_=new std::thread[num_threads];
    hasTasks=new std::condition_variable();
    hasTasksMutex=new std::mutex();
    for (int i=0;i<num_threads;i++){
        threads_pool_[i]=std::thread(&TaskSystemParallelThreadPoolSleeping::Threads_can_sleep,this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    stop=true;
    for(int i=0;i<num_threads_;i++){
        hasTasks->notify_all();
    }
    for(int i=0;i<num_threads_;i++){
        threads_pool_[i].join();
    }
    delete[] threads_pool_;
    delete state_;
    delete hasTasks;
    delete hasTasksMutex;
}

void TaskSystemParallelThreadPoolSleeping::Threads_can_sleep(){
    int id;
    int total;
    while (true){
        if (stop) break;
        state_->mutex_->lock();
        total=state_->num_total_tasks_;
        id=total-state_->left_tasks_;
        if (id<total)state_->left_tasks_--;
        state_->mutex_->unlock();
        if (id<total){
            state_->runnable_->runTask(id, total);
            state_->mutex_->lock();
            state_->finished_tasks_++;
            if (state_->finished_tasks_==total){
                state_->mutex_->unlock();
                state_->finishedMutex_->lock();
                state_->finishedMutex_->unlock();
                state_->finished_->notify_all();

            }
            else {
            state_->mutex_->unlock();
            }
            
        }else {
            std::unique_lock<std::mutex> lk(*hasTasksMutex);
            hasTasks->wait(lk);
            // lk.unlock();
        }
        
    }

};

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::unique_lock<std::mutex> lk(*(state_->finishedMutex_));
    state_->mutex_->lock();
    state_->finished_tasks_ = 0;
    state_->left_tasks_ = num_total_tasks;
    state_->num_total_tasks_ = num_total_tasks;
    state_->runnable_ = runnable;
    state_->mutex_->unlock();
    for (int i = 0; i < num_total_tasks; i++) hasTasks->notify_all();
    // go to sleep, wait for all the tasks to finish
    //std::cerr << "go to sleep" << std::endl;
    state_->finished_->wait(lk); 
    // lk.unlock();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
