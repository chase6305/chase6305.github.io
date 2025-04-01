---
title: '基于三维栅格空间的A*算法流程C++实现'
date: 2022-06-29
lastmod: 2022-06-29
draft: false
tags: ["AI", "Algorithms"]
categories: ["Algorithms"]
authors: ["chase"]
summary: "基于三维栅格空间的A*算法流程C++实现"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

**传统A * 算法介绍**
传统A*算法是一种基于栅格的最短无碰路径求解的启发式算法，其评价函数为：

```math
f(x) = g(x) + h(x)
```

其中：
- g(x): 从起点到当前节点 x 的实际代价
- h(x): 从节点 x 到终点的估计代价（启发值）
- f(x): 通过节点 x 的路径总代价

当所在的环境地图被分割成三维立体地图时，是从起始栅格到栅格x的实际栅格数，是从栅格 x 到目标栅格的预计栅格数，那么是从起始栅格经过中间栅格 x 到达目标栅格的预计栅格数。以下为传统 A * 算法的实现流程（C++）。

**blockallocator.h**
```cpp
#ifndef __BLOCKALLOCATOR_H__
#define __BLOCKALLOCATOR_H__

#include <cstdint>

/// This is a small object allocator used for allocating small
/// objects that persist for more than one time step.
/// See: http://www.codeproject.com/useritems/Small_Block_Allocator.asp
class BlockAllocator{
    static const int kChunkSize = 16 * 1024;
    static const int kMaxBlockSize = 640;
    static const int kBlockSizes = 14;
    static const int kChunkArrayIncrement = 128;

public:
    BlockAllocator();
    ~BlockAllocator();

public:
    void* allocate(int size);
    void free(void *p, int size);
    void clear();

private:
    int             num_chunk_count_;
    int             num_chunk_space_;
    struct Chunk*   chunks_;
    struct Block*   free_lists_[kBlockSizes];
    static int      block_sizes_[kBlockSizes];
    static uint8_t  s_block_size_lookup_[kMaxBlockSize + 1];
    static bool     s_block_size_lookup_initialized_;
};

#endif

```


**blockallocator.cpp**
```cpp
#include "blockallocator.h"
#include <limits.h>
#include <memory.h>
#include <stddef.h>
#include <malloc.h>
#include <assert.h>

struct Chunk{
    int block_size;
    Block *blocks;
};

struct Block{
    Block *next;
};

int BlockAllocator::block_sizes_[kBlockSizes] ={
    16,     // 0
    32,     // 1
    64,     // 2
    96,     // 3
    128,    // 4
    160,    // 5
    192,    // 6
    224,    // 7
    256,    // 8
    320,    // 9
    384,    // 10
    448,    // 11
    512,    // 12
    640,    // 13
};

bool BlockAllocator::s_block_size_lookup_initialized_;

uint8_t BlockAllocator::s_block_size_lookup_[kMaxBlockSize + 1];

BlockAllocator::BlockAllocator(){
    assert(kBlockSizes < UCHAR_MAX);

    num_chunk_space_ = kChunkArrayIncrement;
    num_chunk_count_ = 0;
    chunks_ = (Chunk *)malloc(num_chunk_space_ * sizeof(Chunk));

    memset(chunks_, 0, num_chunk_space_ * sizeof(Chunk));
    memset(free_lists_, 0, sizeof(free_lists_));

    if (s_block_size_lookup_initialized_ == false){
        int j = 0;
        for (int i = 1; i <= kMaxBlockSize; ++i){
            assert(j < kBlockSizes);
            if (i <= block_sizes_[j]){
                s_block_size_lookup_[i] = (uint8_t)j;
            }
            else{
                ++j;
                s_block_size_lookup_[i] = (uint8_t)j;
            }
        }
        s_block_size_lookup_initialized_ = true;
    }
}

BlockAllocator::~BlockAllocator(){
    for (int i = 0; i < num_chunk_count_; ++i){
        ::free(chunks_[i].blocks);
    }
    ::free(chunks_);
}

void* BlockAllocator::allocate(int size){
    if (size == 0){
        return nullptr;
    }

    assert(0 < size);

    if (size > kMaxBlockSize){
        return malloc(size);
    }

    int index = s_block_size_lookup_[size];
    assert(0 <= index && index < kBlockSizes);

    if (free_lists_[index]){
        Block *block = free_lists_[index];
        free_lists_[index] = block->next;
        return block;
    }
    else{
        if (num_chunk_count_ == num_chunk_space_){
            Chunk *oldChunks = chunks_;
            num_chunk_space_ += kChunkArrayIncrement;
            chunks_ = (Chunk *)malloc(num_chunk_space_ * sizeof(Chunk));
            memcpy(chunks_, oldChunks, num_chunk_count_ * sizeof(Chunk));
            memset(chunks_ + num_chunk_count_, 0, kChunkArrayIncrement * sizeof(Chunk));
            ::free(oldChunks);
        }

        Chunk *chunk = chunks_ + num_chunk_count_;
        chunk->blocks = (Block *)malloc(kChunkSize);
#if defined(_DEBUG)
        memset(chunk->blocks, 0xcd, kChunkSize);
#endif
        int block_size = block_sizes_[index];
        chunk->block_size = block_size;
        int block_count = kChunkSize / block_size;
        assert(block_count * block_size <= kChunkSize);
        for (int i = 0; i < block_count - 1; ++i){
            Block *block = (Block *)((uint8_t *)chunk->blocks + block_size * i);
            Block *next = (Block *)((uint8_t *)chunk->blocks + block_size * (i + 1));
            block->next = next;
        }
        Block *last = (Block *)((uint8_t *)chunk->blocks + block_size * (block_count - 1));
        last->next = nullptr;

        free_lists_[index] = chunk->blocks->next;
        ++num_chunk_count_;

        return chunk->blocks;
    }
}

void BlockAllocator::free(void *p, int size){
    if (size == 0 || p == nullptr){
        return;
    }

    assert(0 < size);

    if (size > kMaxBlockSize){
        ::free(p);
        return;
    }

    int index = s_block_size_lookup_[size];
    assert(0 <= index && index < kBlockSizes);

#ifdef _DEBUG
    int block_size = block_sizes_[index];
    bool found = false;
    for (int i = 0; i < num_chunk_count_; ++i)
    {
        Chunk *chunk = chunks_ + i;
        if (chunk->block_size != block_size)
        {
            assert((uint8_t *)p + block_size <= (uint8_t *)chunk->blocks ||
                (uint8_t *)chunk->blocks + kChunkSize <= (uint8_t *)p);
        }
        else
        {
            if ((uint8_t *)chunk->blocks <= (uint8_t *)p && (uint8_t *)p + block_size <= (uint8_t *)chunk->blocks + kChunkSize)
            {
                found = true;
            }
        }
    }

    assert(found);

    memset(p, 0xfd, block_size);
#endif

    Block *block = (Block *)p;
    block->next = free_lists_[index];
    free_lists_[index] = block;
}

void BlockAllocator::clear(){
    for (int i = 0; i < num_chunk_count_; ++i){
        ::free(chunks_[i].blocks);
    }

    num_chunk_count_ = 0;
    memset(chunks_, 0, num_chunk_space_ * sizeof(Chunk));
    memset(free_lists_, 0, sizeof(free_lists_));
}

```
**astar.h**
```cpp
#ifndef __ASTAR_H__
#define __ASTAR_H__

#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

class BlockAllocator;

class AStar
{
public:
    /**
     * 二维向量
     */
    struct Vec3
    {
        uint16_t x;
        uint16_t y;
        uint16_t z;

        Vec3() : x(0) , y(0) , z(0) //初始化
        {
        }

        Vec3(uint16_t x1, uint16_t y1, uint16_t z1) : x(x1), y(y1), z(z1) //初始化赋值列表
        {
        }

        void reset(uint16_t x1, uint16_t y1, uint16_t z1)   //赋值 
        {
            x = x1;
            y = y1;
            z = z1;
        }

        //计算两点之间的距离
        int distance(const Vec3 &other) const
        {
            return abs(other.x - x) + abs(other.y - y) + abs(other.z - z);
        }

        //判断是否等于目标点
        bool operator== (const Vec3 &other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    typedef std::function<bool(const Vec3&)> Callback;

    /**
     * 搜索参数
     */
    struct Params
    {
        bool        corner;     // 允许拐角
        uint16_t    height;     // 地图高度 z
        uint16_t    width;      // 地图宽度 
        uint16_t    depth;      // 地图深度 z
        Vec3        start;      // 起点坐标
        Vec3        end;        // 终点坐标
        Callback    can_pass;   // 是否可通过

        Params() : height(0), width(0), depth(0), corner(false)
        {
        }
    };

private:
    /**
     * 路径节点状态
     */
    enum NodeState
    {
        NOTEXIST,               // 不存在
        IN_OPENLIST,            // 在开启列表
        IN_CLOSEDLIST           // 在关闭列表
    };

    /**
     * 路径节点
     */
    struct Node
    {
        uint16_t    g;          // 与起点距离
        uint16_t    h;          // 与终点距离
        Vec3        pos;        // 节点位置
        NodeState   state;      // 节点状态
        Node*       parent;     // 父节点

        /**
         * 计算f值
         */
        int f() const
        {
            return g + h;
        }

        inline Node(const Vec3 &pos)
            : g(0), h(0), pos(pos), parent(nullptr), state(NOTEXIST)
        {
        }
    };

public:
    AStar(BlockAllocator *allocator);

    ~AStar();

public:
    /**
     * 获取直行估值
     */
    int get_step_value() const;

    /**
     * 获取拐角估值
     */
    int get_oblique_value() const;

    /**
     * 设置直行估值
     */
    void set_step_value(int value);

    /**
     * 获取拐角估值
     */
    void set_oblique_value(int value);

    /**
     * 执行寻路操作
     */
    std::vector<Vec3> find(const Params &param);

private:
    /**
     * 清理参数
     */
    void clear();

    /**
     * 初始化参数
     */
    void init(const Params &param);

    /**
     * 参数是否有效
     */
    bool is_vlid_params(const Params &param);

private:
    /**
     * 二叉堆上滤
     */
    void percolate_up(size_t hole);

    /**
     * 获取节点索引
     */
    bool get_node_index(Node *node, size_t *index);

    /**
     * 计算G值
     */
    uint16_t calcul_g_value(Node *parent, const Vec3 &current);

    /**
     * 计算F值
     */
    uint16_t calcul_h_value(const Vec3 &current, const Vec3 &end);

    /**
     * 节点是否存在于开启列表
     */
    bool in_open_list(const Vec3 &pos, Node *&out_node);

    /**
     * 节点是否存在于关闭列表
     */
    bool in_closed_list(const Vec3 &pos);

    /**
     * 是否可通过
     */
    bool can_pass(const Vec3 &pos);

    /**
     * 当前点是否可到达目标点
     */
    bool can_pass(const Vec3 &current, const Vec3 &destination, bool allow_corner);

    /**
     * 查找附近可通过的节点
     */
    void find_can_pass_nodes(const Vec3 &current, bool allow_corner, std::vector<Vec3> *out_lists);

    /**
     * 处理找到节点的情况
     */
    void handle_found_node(Node *current, Node *destination);

    /**
     * 处理未找到节点的情况
     */
    void handle_not_found_node(Node *current, Node *destination, const Vec3 &end);

private:
    int                     step_val_;          //直行估值
    int                     oblique_val_;       //拐角估值
    std::vector<Node*>      mapping_;           //排列？？？  如何将三维空间的所有点按一定顺序去排列 索引号均大于零
    uint16_t                height_;            //
    uint16_t                width_;             //
    uint16_t                depth_;             //
    Callback                can_pass_;          //
    std::vector<Node*>      open_list_;         //开启列表
    BlockAllocator*         allocator_;         //寻路者
};

#endif

```

**astar.cpp**
```cpp
#include "astar.h"
#include <cassert>
#include <cstring>
#include <algorithm>
#include "blockallocator.h"

//直行估值和拐角估值
static const int kStepValue = 10;       //水平或垂直移动的耗费为10
static const int kObliqueValue = 14;    //对角线移动的耗费为14 

//构造函数
AStar::AStar(BlockAllocator *allocator) 
    : width_(0)
    , height_(0)
    , depth_(0)
    , allocator_(allocator)
    , step_val_(kStepValue)
    , oblique_val_(kObliqueValue)
{
    assert(allocator_ != nullptr);
}

//析构函数
AStar::~AStar(){
    clear();
}

// 获取直行估值
int AStar::get_step_value() const{
    return step_val_;
}

// 获取拐角估值
int AStar::get_oblique_value() const{
    return oblique_val_;
}

// 设置直行估值
void AStar::set_step_value(int value){
    step_val_ = value;
}

// 获取拐角估值
void AStar::set_oblique_value(int value){
    oblique_val_ = value;
}

// 析构函数---清理参数
void AStar::clear(){
    size_t index = 0;
    const size_t max_size = width_ * height_ * depth_ ; //设置地图大小
    while (index < max_size)  {
        allocator_->free( mapping_[index++] , sizeof(Node) ); //
    }
    open_list_.clear(); //清空openglist
    can_pass_ = nullptr; //空
    width_ = height_ = depth_ = 0;  //清零
}

// 初始化操作
void AStar::init(const Params &param)
{
    width_ = param.width;
    height_ = param.height;
    depth_ = param.depth;
    can_pass_ = param.can_pass;
    //如果地图不为空，则清空
    if (!mapping_.empty()) {
        memset( &mapping_[0] , 0, sizeof(Node*) * mapping_.size() ); //初始化 地图清空为0 
    }
    mapping_.resize( width_ * height_ * depth_ , nullptr); //调整大小
}

// 搜索参数是否有效-----建立三维数组 （  ）
// 设置搜索范围width height depth 都大于零
// 起点和终点都大于零，且在搜索范围内
bool AStar::is_vlid_params(const AStar::Params &param){
    return (param.can_pass != nullptr
            && (param.width > 0 && param.height > 0 && param.depth > 0 )
            && (param.end.x >= 0 && param.end.x < param.width)
            && (param.end.y >= 0 && param.end.y < param.height)
            && (param.end.z >= 0 && param.end.z < param.depth)
            && (param.start.x >= 0 && param.start.x < param.width)
            && (param.start.y >= 0 && param.start.y < param.height)
            && (param.start.z >= 0 && param.start.z < param.depth)
            );
}

// 获取节点索引
bool AStar::get_node_index(Node *node, size_t *index){
    *index = 0;
    const size_t size = open_list_.size();
    while (*index < size)
    {
        if (open_list_[*index]->pos == node->pos)
        {
            return true;
        }
        ++(*index);
    }
    return false;
}

// 二叉堆上滤
void AStar::percolate_up(size_t hole){
    size_t parent = 0;
    while (hole > 0){
        parent = (hole - 1) / 2; //父节点位于hole/2的位置
        if ( open_list_[hole]->f() < open_list_[parent]->f() ){
            std::swap( open_list_[hole] , open_list_[parent] );
            hole = parent;
        }
        else{
            return;
        }
    }
}

// 计算G值
inline uint16_t AStar::calcul_g_value(Node *parent, const Vec3 &current){
    uint16_t g_value = current.distance(parent->pos) == 2 ? oblique_val_ : step_val_;
    return g_value += parent->g;
}

// 计算F值
inline uint16_t AStar::calcul_h_value(const Vec3 &current, const Vec3 &end){
    unsigned int h_value = end.distance(current);
    return h_value * step_val_;
}

// 节点是否存在于开启列表
inline bool AStar::in_open_list(const Vec3 &pos, Node *&out_node){
    out_node = mapping_[ pos.z * width_ * height_ + pos.y * width_ + pos.x ];    
    return out_node ? out_node->state == IN_OPENLIST : false;
}

// 节点是否存在于关闭列表
inline bool AStar::in_closed_list(const Vec3 &pos){
    Node *node_ptr = mapping_[pos.z * width_ * height_ + pos.y * width_ + pos.x];  
    return node_ptr ? node_ptr->state == IN_CLOSEDLIST : false;
}

// 是否可到达//点是否在可达空间内
bool AStar::can_pass(const Vec3 &pos) {
    return (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_ && pos.z >= 0 && pos.z < depth_ ) ? can_pass_(pos) : false;
}

// 当前点是否可到达目标点
bool AStar::can_pass(const Vec3 &current, const Vec3 &destination, bool allow_corner){
    if (destination.x >= 0 && destination.x < width_ 
    && destination.y >= 0 && destination.y < height_ 
    && destination.z >= 0 && destination.z < depth_ ) {
        //该点是否在关闭列表中
        if (in_closed_list(destination)){
            return false;
        }
        //destination 和 current 两点之间距离是否等于1
        if (destination.distance(current) == 1){
            return can_pass_(destination);      //判断该点是否可到达
        }
         // 允许转角
        else if (allow_corner){
            return can_pass_(destination) 
                    && (can_pass(Vec3( destination.x , current.y ,      current.z )) 
                    && can_pass(Vec3(  current.x,      destination.y  , current.z))
                    && can_pass(Vec3(  current.x,      current.y  ,     destination.z))); 
        }
    }
    return false;
}

// 查找附近可通过的节点
// 输入当前节点 做三维
void AStar::find_can_pass_nodes(const Vec3 &current, bool corner, std::vector<Vec3> *out_lists){
    Vec3 destination;   //新建三维点
    const int max_row = current.y + 1;  //
    const int max_col = current.x + 1;  // 
    const int max_z = current.z + 1; // 

    int row_index = current.y - 1;  // row 为纵向


    int z_index = current.z - 1 ;   // z 为层数
    //若小于零，防止超过下限
    if ( z_index < 0 ){
        z_index = 0 ;
    }
    while ( z_index <= max_z ){
        int row_index = current.z - 1 ; 
        //若row_index小于零，则将row_index置为零 以防超过边界
        if (row_index < 0)  {
            row_index = 0;
        }
        while ( row_index <= max_row){
            int col_index = current.x - 1 ;
            //若col_index小于零，则将col_index置为零 以防超过边界
            if ( col_index < 0 ){
                col_index = 0 ;
            }
            //若max_col 大于等于 col_index 
            while (col_index <= max_col){
                destination.reset( col_index, row_index, z_index );    //将当前点节点信息存放到destinationn  
                //判断是否到达终点
                if (can_pass(current, destination, corner)){
                    out_lists->push_back(destination);
                }
                ++col_index;
            }
            ++row_index;
        }
        ++z_index;
    }

}

// 处理找到节点的情况
void AStar::handle_found_node(Node *current, Node *destination){
    unsigned int g_value = calcul_g_value(current, destination->pos); //计算g值 
    if (g_value < destination->g){
        destination->g = g_value;
        destination->parent = current;

        size_t index = 0;
        //处理找到节点的情况
        if (get_node_index(destination, &index)){
            percolate_up(index);    //二叉堆上滤
        }
        else{
            assert(false);
        }
    }
}

// 处理未找到节点的情况
void AStar::handle_not_found_node(Node *current, Node *destination, const Vec3 &end){
    destination->parent = current;
    destination->h = calcul_h_value(destination->pos, end);     //计算h值 
    destination->g = calcul_g_value(current, destination->pos); //计算g值 

    Node *&reference_node = mapping_[destination->pos.z * width_ * height_ + destination->pos.y * width_ + destination->pos.x];
    reference_node = destination;
    reference_node->state = IN_OPENLIST;

    open_list_.push_back(destination);
    std::push_heap(open_list_.begin(), open_list_.end(), [](const Node *a, const Node *b)->bool {
        return a->f() > b->f();
    });
}

// 执行寻路操作
// 传入 param 搜索参数
// 得到最终路径结果
std::vector<AStar::Vec3> AStar::find(const Params &param){
    std::vector<Vec3> paths; //路径
    assert(is_vlid_params(param)); //检查参数是否有效
    if (!is_vlid_params(param)){
        return paths;
    }

    // 初始化
    init(param);  //初始化参数并分配地图大小
    std::vector<Vec3> nearby_nodes;   //创建附近节点的容器
    nearby_nodes.reserve(param.corner ? 26 : 6 );  //分配容器大小， 二维 8 或 4  三维 26 或 6 ----corner：是否允许拐角

    // 将起点放入开启列表
    Node *start_node = new( allocator_->allocate( sizeof(Node) ) ) Node(param.start); 
    open_list_.push_back(start_node);
    //depth_  = ?  是等于 width_ * height_ 
    Node *&reference_node = mapping_[ start_node->pos.z * width_ * height_  + start_node->pos.y * width_ + start_node->pos.x ]; //创建路径节点
    reference_node = start_node;        //将起点放入reference_node中
    reference_node->state = IN_OPENLIST;   //将起点节点状态设置为开启，起点放入开启列表 

    // 寻路操作 遍历开启列表直到找到终点 
    while (!open_list_.empty()){
        // 找出f值最小节点NumberOfGrid_x
        Node *current = open_list_.front(); //返回当前vector容器中起始元素的引用
        std::pop_heap(open_list_.begin(), open_list_.end(), [](const Node *a, const Node *b)->bool{
            return a->f() > b->f();
        }); // 将堆顶元素调整到最后
        open_list_.pop_back(); //弹出末尾元素
        mapping_[ current->pos.z * width_ * height_ + current->pos.y * width_ + current->pos.x ]->state = IN_CLOSEDLIST; //将节点放进关闭列表 （起点）

        // 是否找到终点    //重载==号 判断两个的坐标是否相等。
        if (current->pos == param.end){
            //寻找到终点时，将此路径上的父节点以及子节点保存到paths中
            while (current->parent){
                paths.push_back(current->pos);
                current = current->parent;
            }
            std::reverse(paths.begin(), paths.end());  //反转操作
            goto __end__;
        }
        // 查找周围可通过节点
        nearby_nodes.clear(); //清空nearby_nodes，将新节点周围可通过的节点放入nearby_nodes
        find_can_pass_nodes(current->pos, param.corner, &nearby_nodes);  //找寻当前节点的附近可通过节点 
        //current->pos 当前路径节点的节点位置 
        // 计算周围节点的估值
        size_t index = 0;
        const size_t size = nearby_nodes.size(); 
        while (index < size){
            Node *next_node = nullptr; 
            //判断节点是否在开启列表
            if ( in_open_list(nearby_nodes[index], next_node) ){
                handle_found_node(current, next_node);    //处理找到节点的情况
            }
            else{
                next_node = new(allocator_->allocate(sizeof(Node))) Node(nearby_nodes[index]);  
                handle_not_found_node(current, next_node, param.end);   //处理未找到节点的情况
            }
            ++index;
        }
    }
__end__:
    clear();
    return paths;
}

```

**PathPlanComposition.h**
```cpp
/*
 * @Author: JT
 * @Date: 2020-06-18 15:27:57
 * @LastEditTime: 2020-08-10 13:15:05
 * @LastEditors: JT
 * @Description: In User Settings Edit
 */ 
#ifndef PATHPLANINGCOMPOSITION_H_
#define PATHPLANINGCOMPOSITION_H_
#include<iostream>
#include "data.h"
#include "HeadFile.h"

class PathPlanComposition{
private:
    /* data */
public:  
    PathPlanComposition(/* args */);
    ~PathPlanComposition();
    
    void AstarRoute() ;
};

PathPlanComposition::PathPlanComposition(/* args */){
}

PathPlanComposition::~PathPlanComposition(){
}


#endif
```


**PathPlanComposition.cpp**
```cpp

#include "PathPlanComposition.h"
#include "astar.cpp"
#include "blockallocator.cpp"

// Astar算法 示例程序 
// 可自行构建三维栅格地图
// 在param.start和param.end传入三维栅格的起点和终点，此处第三个值为0 默认为平面
void PathPlanComposition::AstarRoute( ){
	//地图
	char maps[10][10] =
	{
		{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 0, 1, 0, 1, 0, 1 },
		{ 1, 1, 1, 1, 0, 1, 0, 1, 0, 1 },
		{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 1 },
		{ 0, 1, 0, 1, 1, 1, 1, 1, 0, 1 },
		{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
		{ 1, 1, 0, 0, 1, 0, 1, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
	};
	AStar::Params param; 
	param.width = 10;	//x	
	param.height = 10;	//y	
	param.depth = 1;	//z	
	param.corner = false;
	param.start = AStar::Vec3( 9 , 9 , 0 );	//起点
	param.end 	= AStar::Vec3( 1 , 1 , 0 ); //终点
	param.can_pass = [&](const AStar::Vec3 &pos)->bool{
		return maps[pos.y][pos.x] == 0;
	};

	// 执行搜索
	BlockAllocator allocator;
	AStar algorithm(&allocator);
	auto path = algorithm.find(param);
	if( path.empty()){	
		cout << " path empty" << endl; 
	}
	for( int i = 0 ; i< path.size() ; i++ ){
		cout << " path-------- " << path[i].x << " , " << path[i].y << " , " << path[i].z << endl;
	}

}
```
