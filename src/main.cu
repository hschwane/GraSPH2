/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#include <thrust/random.h>
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include <cuda_gl_interop.h>
#include <cmath>
#include <initialConditions/particleSources/PlummerSphere.h>
#include <bitset>
#include <iterator>

#include "tbb/tbb.h"

#include "initialConditions/InitGenerator.h"
#include "initialConditions/particleSources/UniformSphere.h"
#include "initialConditions/particleSources/TextFile.h"
#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "sph/kernel.h"
#include "sph/eos.h"
#include "sph/models.h"
#include "ResultStorageManager.h"
#include "settings.h"

// compile setting files into resources
ADD_RESOURCE(Settings,"settings.h");
ADD_RESOURCE(HeadlessSettings,"headlessSettings.h");
ADD_RESOURCE(PrecisionSettings,"precisionSettings.h");

//constexpr f1_t H2 = H*H; //!< square of the smoothing length
//constexpr f1_t dW_prefactor = kernel::dsplinePrefactor<dimension>(H); //!< spline kernel prefactor
//constexpr f1_t W_prefactor = kernel::splinePrefactor<dimension>(H); //!< spline kernel prefactor




using spaceKey = unsigned long long int;
/**
 * @brief Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit.
 *          https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
 * @param x integer to expand
 * @return expanded integer
 */
CUDAHOSTDEV unsigned long long int expandBits( unsigned long long int x)
{
    x = (x | x << 32u) & 0x1f00000000ffffu;
    x = (x | x << 16u) & 0x1f0000ff0000ffu;
    x = (x | x << 8u) & 0x100f00f00f00f00fu;
    x = (x | x << 4u) & 0x10c30c30c30c30c3u;
    x = (x | x << 2u) & 0x1249249249249249u;
    return x;
}

/**
 * @brief Calculates a 64-bit Morton code for the given 3D point located within the unit cube [0,1].
 *          https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/ on april 29 2019
 * @param x first spacial coordinate
 * @param y second spacial coordinate
 * @param z third spacial coordinate
 * @return resulting 64 bit morton code
 */
CUDAHOSTDEV spaceKey mortonKey(f1_t x, f1_t y, f1_t z)
{
    // float in [0,1] to 21 bit integer
    x = min(max(x * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    y = min(max(y * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    z = min(max(z * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    spaceKey xx = expandBits(static_cast<spaceKey>(x));
    spaceKey yy = expandBits(static_cast<spaceKey>(y));
    spaceKey zz = expandBits(static_cast<spaceKey>(z));
    return xx | (yy << 1u) | (zz << 2u);
}

CUDAHOSTDEV spaceKey calculatePositionKey(const f3_t& pos, const f3_t& domainMin, const f3_t& domainFactor)
{
    const f3_t normalizedPos = (pos - domainMin) * domainFactor;
    return mortonKey(normalizedPos.x,normalizedPos.y,normalizedPos.z);
}

// tree settings
/////////////////
constexpr f3_t domainMin = {-2,-2,-2};
constexpr f3_t domainMax = {2,2,2};
constexpr f1_t theta = 0.6_ft;
constexpr f1_t eps2 = 0.001_ft;
constexpr int maxLeafParticles = 16; // max number of particles in a leaf
constexpr int maxCriticalParticles = 32; // max number of particles in a critical node (critical nodes traverse the tree together)

// gpu settings
constexpr int gpuBlockSize = 32; // size of one cuda block
constexpr int stackSizePerBlock = 4096; // global memory available to each thread block to use as a stack

// cpu settings
constexpr int stackSizePerThreadCPU = 4096;
constexpr int interactionListSizeCPU = 8*64000;
constexpr int nodeListSizeCPU = 8 * 1000;
//#define DEBUG_PRINTS
/////////////////

// NOTE: this limits N to 134'217'728 particles
// and allows for 1 to 16 children (a node with no children is not stored)
struct MPU_ALIGN(4) TreeDownlink
{
public:

    TreeDownlink() : m_data(0) {}
    TreeDownlink(unsigned int nchildren, unsigned int childid, bool isLeaf) : m_data(((unsigned int)(isLeaf) << leafShift) | ((nchildren-1) << numShift) | (childid & firstMask)) {}
    CUDAHOSTDEV void setData(unsigned int nchildren, unsigned int childid, bool isLeaf) {m_data = ((unsigned int)(isLeaf) << leafShift) | ((nchildren-1) << numShift) | (childid & firstMask); }
    CUDAHOSTDEV void setIsLeaf(bool isLeaf) { m_data = (m_data & ~(leafMask)) | ((unsigned int)(isLeaf) << leafShift);}
    CUDAHOSTDEV void setNChildren(unsigned int nchildren) { m_data = (m_data & ~(numMask)) | ((nchildren-1) << numShift);} //!< if you put in a number outside of [1,16] the universe might explode!
    CUDAHOSTDEV void setFirstChild(unsigned int childid) { m_data = (m_data & ~(firstMask)) | (childid & firstMask);}

    CUDAHOSTDEV unsigned int firstChild() const { return (m_data & firstMask); }
    CUDAHOSTDEV unsigned int nChildren() const { return ((m_data & numMask) >> 27u) +1; }
    CUDAHOSTDEV bool isLeaf() const { return (m_data & leafMask) != 0;}

private:
    static constexpr unsigned int leafShift = 31u;
    static constexpr unsigned int leafMask = (1u<<leafShift); // 10000000 00000000 00000000 00000000
    static constexpr unsigned int numShift = 27u;
    static constexpr unsigned int numMask = (15u<<numShift);  // 01111000 00000000 00000000 00000000
    static constexpr unsigned int firstMask = ~(31u<<27u);    // 00000111 11111111 11111111 11111111
    unsigned int m_data;
};

struct MPU_ALIGN(16) NodeOpeningData
{
    f3_t com;
    f1_t od2;
};

struct MPU_ALIGN(16) CriticalNode
{
    int firstParticle;
    int nParticles;
    int nodeId;
};

CUDAHOSTDEV f1_t calcOpeningDistance2(const f3_t& com, const f3_t& min, const f3_t& max)
{
    f3_t l3d = max - min;
    f1_t l = fmax(fmax(l3d.x,l3d.y),l3d.z);
    f3_t cogeo = min + (l3d*0.5_ft);
    f1_t delta = length(com-cogeo);
    f1_t od = l / theta + delta;
    f1_t od2 = od*od;
    return (std::isnan(od2) ? std::numeric_limits<f1_t>::infinity() : od2);
}

CUDAHOSTDEV f3_t calcInteraction( f1_t massj, f1_t r2, f3_t rij)
{
    // gravity
    const f1_t distSqr = r2 + eps2;
    const f1_t invDist = rsqrt(distSqr);
    const f1_t invDistCube = invDist * invDist * invDist;
    return -rij * massj * invDistCube;
}

// TODO: fix distance calculation
CUDAHOSTDEV f1_t minDistanceSquare(f3_t min, f3_t max, f3_t point)
{
    const f3_t dmin = fabs(min-point);
    const f3_t dmax = fabs(max-point);
    const f3_t mdv = fmin(dmin,dmax);
    return dot(mdv,mdv);
}

template <typename DeviceBufferReference>
__global__ void traverseTreeGPU(DeviceBufferReference pb,
                                CriticalNode* critNodes,
                                f3_t* aabbMin,
                                f3_t* aabbMax,
                                TreeDownlink* links,
                                NodeOpeningData* openingData,
                                f1_t* nodeMass,
                                int* globalStack)
{
    // setup cub
    typedef cub::BlockScan<int, gpuBlockSize> BlockScan;
    typedef cub::BlockRadixSort<unsigned int, gpuBlockSize, 1, int> BlockRadixSort;
    typedef cub::BlockDiscontinuity<int, gpuBlockSize> BlockDiscontinuity;
//    __shared__ union
//    {
        __shared__ typename BlockScan::TempStorage blockScan;
        __shared__ typename BlockRadixSort::TempStorage radixSort;
        __shared__ typename BlockDiscontinuity::TempStorage discontinutiy;
//    } cubTempData;

    // allocate shared memory
    constexpr int pilSize = gpuBlockSize * 16;
    constexpr int sharedStackSize = gpuBlockSize * 8;
    constexpr int nilSize = gpuBlockSize;

    enum OpeningType : unsigned int
    {
        self = 3,
        particleInteraction = 2,
        nodeInteraction = 1,
        nodeToStack = 0
    };

    // shared memory for temporary interaction list and stack storage
    __shared__ int pil[pilSize];                                      // blockSize * 16 * 4 = 16384 byte
    __shared__ int sharedStack[sharedStackSize];                      // blockSize * 8 * 4 = 8192 byte
    __shared__ int nil[nilSize];                                      // blockSize * 4 = 1024
    __shared__ int* bufferPointer[4];
    __shared__ int accumEntryCounts[4];
    __shared__ int entryCounts[4];

    // shared memory for acceleration of interaction
    SharedParticles<gpuBlockSize,SHARED_POSM> sharedParticleData; // blockSize * 16 = 4096 + ?

    // stack pointer management
    __shared__ int* stackWriteEnd;
    __shared__ int* stackReadEnd;
    __shared__ int* stackWrite;
    __shared__ int* stackRead;

    // data about our group
    __shared__ CriticalNode group;                   // 16
    __shared__ f3_t groupMin;                        // 16
    __shared__ f3_t groupMax;                        // 16

    // load values for this critical node / this thread group
    if(threadIdx.x == 0)
    {
        group = critNodes[blockIdx.x];
        groupMin = aabbMin[group.nodeId];
        groupMax = aabbMax[group.nodeId];

        accumEntryCounts[0] = 0;
        accumEntryCounts[1] = 0;
        accumEntryCounts[2] = 0;
        accumEntryCounts[3] = 0;

        bufferPointer[OpeningType::self] = nullptr;
        bufferPointer[OpeningType::particleInteraction] = &pil[0];
        bufferPointer[OpeningType::nodeInteraction] = &nil[0];
        bufferPointer[OpeningType::nodeToStack] = &sharedStack[0];

        stackReadEnd = &globalStack[blockIdx.x * stackSizePerBlock];
        stackWriteEnd = stackReadEnd + stackSizePerBlock / 2;

//        printf("stack read address: %i \n", stackReadEnd);
//        printf("stack write address: %i \n", stackWriteEnd);

        stackWrite = stackWriteEnd;
        stackRead = stackReadEnd;

        // load children of root into stack
        for(int child = links[0].firstChild(); child < links[0].firstChild() + links[0].nChildren(); child++)
            *(stackWrite++) = child;
        // TODO: initialize stack more efficient and with more nodes or do it before and store in constant memory (eg start at second level)

        // swap stack pointer
        stackRead = stackWrite;
        stackWrite = stackReadEnd;
        stackReadEnd = stackWriteEnd;
        stackWriteEnd = stackWrite;
    }
    __syncthreads();

    // load this threads particle
    Particle<POS,MASS,ACC> pi{};
    if(threadIdx.x < group.nParticles)
        pi = pb.template loadParticle<POS,MASS>(group.firstParticle + threadIdx.x); // cuda block size size MUST be bigger than the critical node size

    // each loop iteration processes one tree layer
    // traverse is finished when read stack is empty directly after stack swapping
    int layer  =1;
    while(stackRead != stackReadEnd)
    {
//        if(threadIdx.x == 0)
//            printf("layer %i ------------------------------------\n----------------------------------------------------\n",layer++);
        // each loop iteration processes block size elements on the stack
        // until stack is empty
        const int numNodes = stackRead - stackReadEnd;
        for(int iteration = 0; iteration < numNodes; iteration+= blockDim.x)
        {
            // load the node id from the stack (in global memory)
            const int i = iteration + threadIdx.x;
            int id = (i < numNodes) ? *((stackRead - 1) - i) : group.nodeId;

//            printf("i: %i, id: %i \n", i, id);

            // figure out type of interaction
            const NodeOpeningData od = openingData[id];
            TreeDownlink link = links[id];
            f1_t r2 = minDistanceSquare(groupMin, groupMax, od.com);
            OpeningType opening;
//            = (group.nodeId == id) ? OpeningType::self
//                                                       :  (r2 > od.od2) ? OpeningType::nodeInteraction
//                                                                                    : (link.isLeaf())
//                                                                                      ? OpeningType::particleInteraction
//                                                                                      : OpeningType::nodeToStack;

            if(group.nodeId == id)
                opening = OpeningType::self;
            else if(r2 <= od.od2)
            {
                if(link.isLeaf())
                    opening = OpeningType::particleInteraction;
                else
                    opening = OpeningType::nodeToStack;
            }
            else
                opening = OpeningType::nodeInteraction;


//            printf("i: %i, id: %i, opening: %i \n", i, id, opening);
//            if(i==1 && id == 6)
//            {
//                printf("opening distance: %f, com: %f - %f - %f \n", od.od2, od.com.x, od.com.y, od.com.z);
//            }

            // sort depending on opening type
            BlockRadixSort(radixSort).Sort( reinterpret_cast<unsigned int (&)[1]>(opening),
                                            reinterpret_cast<int (&)[1]>(id));
            link = links[id];

//            printf("i: %i, id: %i, opening: %i \n", i, id, opening);

            // use prefix sum to determin where to write in shared memory
            int numToWrite = (opening == OpeningType::self) ? 0
                                                            : (opening == OpeningType::nodeInteraction) ? 1
                                                                                                        : link.nChildren();
            int writeID;
            BlockScan(blockScan).ExclusiveSum(numToWrite, writeID);

            // we also need to know how many entries of each opening type will be in storage
            int flag;
            BlockDiscontinuity(discontinutiy).FlagTails( reinterpret_cast<int (&)[1]>(flag), reinterpret_cast<int (&)[1]>(opening), cub::Inequality());

//            printf("i: %i, id: %i, opening: %i, numToWrite: %i, writeID: %i, flag: %i \n", i, id, opening, numToWrite, writeID, flag);

            if(flag == 1 && opening < 3)
            {
                accumEntryCounts[opening] = writeID+numToWrite;
            }

            __syncthreads();

            if(threadIdx.x == 0)
            {
                if(accumEntryCounts[1] == 0)
                    accumEntryCounts[1] = accumEntryCounts[0];
                if(accumEntryCounts[2] == 0)
                    accumEntryCounts[2] = accumEntryCounts[1];

                entryCounts[0] = accumEntryCounts[0];
                entryCounts[1] = accumEntryCounts[1] - accumEntryCounts[0];
                entryCounts[2] = accumEntryCounts[2] - accumEntryCounts[1];

//                printf("accumulated interaction counts for type 0: %i\n", accumEntryCounts[0]);
//                printf("accumulated interaction counts for type 1: %i\n", accumEntryCounts[1]);
//                printf("accumulated interaction counts for type 2: %i\n", accumEntryCounts[2]);
            }

            __syncthreads();

            // handle stack if needed
            int currentSharedStacksize = bufferPointer[OpeningType::nodeToStack] - &sharedStack[0];
            if(entryCounts[OpeningType::nodeToStack] + currentSharedStacksize > sharedStackSize )
            {
                int stackCopyCount = bufferPointer[OpeningType::nodeToStack] - &sharedStack[0];
                for(int j : mpu::blockStrideRange(stackCopyCount))
                {
                    *(stackWrite + j) = sharedStack[j];
                }

                if(threadIdx.x == 0)
                {
                    stackWrite += stackCopyCount;
                    bufferPointer[OpeningType::nodeToStack] = &sharedStack[0];
                }
            }

            // handle node interations if needed
            int currentNilSize = bufferPointer[OpeningType::nodeInteraction] - &nil[0];
            if(entryCounts[OpeningType::nodeInteraction] + currentNilSize > nilSize)
            {
                // TODO: use shared memory for this
                for(int x = 0; x<currentNilSize; x++)
                {
                    int j = nil[x];
                    f3_t rij = pi.pos - openingData[j].com;
                    pi.acc += calcInteraction(nodeMass[j], dot(rij,rij),rij);
                }
                if(threadIdx.x == 0)
                {
                    bufferPointer[OpeningType::nodeInteraction] = &nil[0];
                }
            }

            // handle particle interations if needed
            int currentPilSize = bufferPointer[OpeningType::particleInteraction] - &pil[0];
            if(entryCounts[OpeningType::particleInteraction] + currentPilSize > pilSize)
            {
                // TODO: use shared memory for this
                for(int x = 0; x<currentPilSize; x++)
                {
                    auto pj = pb.template loadParticle<POS,MASS>(pil[x]);
                    f3_t rij = pi.pos - pj.pos;
                    pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
                }
                if(threadIdx.x == 0)
                {
                    bufferPointer[OpeningType::particleInteraction] = &pil[0];
                }
            }

            // --------------------------------------
            // now write the data to the intermediate buffers

            // figure out memory adresss where to write
            __syncthreads();
            int* write = bufferPointer[opening];
            write += (writeID - ((opening == 0) ? 0 : accumEntryCounts[opening - 1]));

//            printf("i: %i, numToWrite: %i, writeID: %i, write: %p, sharedStack: %p \n", i, numToWrite, writeID, write, &sharedStack[0]);
            __syncthreads();

            // write
            int data = (opening == OpeningType::nodeInteraction) ? id : link.firstChild();
            for(int j = 0; j < numToWrite; j++)
            {
//                printf("write to shared addr: %p id: %i\n", &write[j], data+j );
                write[j] = data+j;
            }

            // update the buffer pointer for next iteration
            if(threadIdx.x <3)
            {
                bufferPointer[threadIdx.x] += entryCounts[threadIdx.x];
                accumEntryCounts[threadIdx.x] = 0;
            }
            __syncthreads();
        }

        __syncthreads();

        // store remaining items from shared memory stack to the global memory stack
        int stackCopyCount = bufferPointer[OpeningType::nodeToStack] - &sharedStack[0];

//        if(threadIdx.x == 0)
//            printf("writing %i elments to stack\n",stackCopyCount);

        for(int j : mpu::blockStrideRange(stackCopyCount))
        {
            *(stackWrite + j) = sharedStack[j];
        }

        if(threadIdx.x == 0)
        {
            stackWrite += stackCopyCount;
            bufferPointer[OpeningType::nodeToStack] = &sharedStack[0];
        }

        // swap stack
        if(threadIdx.x == 0)
        {
            stackRead = stackWrite;
            stackWrite = stackReadEnd;
            stackReadEnd = stackWriteEnd;
            stackWriteEnd = stackWrite;
        }
        __syncthreads();
    }


    // --------------------------------------
    // traverse is done, handle all interactions left in shared memory buffers


//    if(threadIdx.x < group.nParticles)
//        printf("p %i acc: %f , %f , %f \n", group.firstParticle + threadIdx.x, pi.acc.x, pi.acc.y, pi.acc.z);

    // handle left over node interations if needed
    int currentNilSize = bufferPointer[OpeningType::nodeInteraction] - &nil[0];
//    if(threadIdx.x == 0)
//        printf("handle %i node interactions \n",currentNilSize);
    // TODO: use shared memory for this
    for(int x = 0; x<currentNilSize; x++)
    {
        int j = nil[x];
        f3_t rij = pi.pos - openingData[j].com;
        pi.acc += calcInteraction(nodeMass[j], dot(rij,rij),rij);
    }

    // handle left over particle interations if needed
    int currentPilSize = bufferPointer[OpeningType::particleInteraction] - &pil[0];
//    if(threadIdx.x == 0)
//        printf("handle %i particle interactions \n",currentPilSize);
    // TODO: use shared memory for this
    for(int x = 0; x<currentPilSize; x++)
    {
        auto pj = pb.template loadParticle<POS,MASS>(pil[x]);
        f3_t rij = pi.pos - pj.pos;
        pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
    }

    // handle self interactions inside the group
    sharedParticleData.storeParticle(threadIdx.x, pi);
    __syncthreads();
    for(int j = 0; j < group.nParticles; j++)
    {
        auto pj = sharedParticleData.loadParticle<POS,MASS>(j);
        f3_t rij = pi.pos - pj.pos;
        pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
    }

//    if(threadIdx.x < group.nParticles)
//        printf("p %i acc: %f , %f , %f \n", group.firstParticle + threadIdx.x, pi.acc.x, pi.acc.y, pi.acc.z);

    // --------------------------------------
    // store acceleration (for now discard results of threads with no particle)
    if(threadIdx.x < group.nParticles)
        pb.storeParticle( group.firstParticle + threadIdx.x, Particle<ACC>(pi));
}

class HostTree
{

public:
    //!< this is not const since particles will be sorted
    template <typename BufferType>
    void construct(BufferType& pb)
    {
        mpu::HRStopwatch sw;

        calcMortonCodes(pb);
        sortParticles(pb);
        generateNodes(pb);
        linkNodes();

        sw.pause();
        std::cout << "Total time of tree construction " << sw.getSeconds() *1000 << "ms" << std::endl;
    }

    template <typename BufferType>
    void update(BufferType& pb)
    {
        mpu::HRStopwatch sw;

        for(int i = m_layerId.size() - 2; i >= 0; i-- )
        {
            // for each layer loop over all nodes in it
            int layerStart = m_layerId[i];
            int layerEnd = m_layerId[i + 1];
            for(int n = layerStart; n < layerEnd; n++)
            {
                // for each node go over all leafs to calculate com, total mass and bounding box
                f3_t min = domainMax;
                f3_t max = domainMin;
                f3_t com{0,0,0};
                f1_t mass=0;

                if(m_links[n].isLeaf())
                {
                    for(int c = m_links[n].firstChild(); c < m_links[n].firstChild() + m_links[n].nChildren(); c++)
                    {
                        auto p = pb.template loadParticle<POS,MASS>(c);
                        min = fmin(min, p.pos);
                        max = fmax(max, p.pos);
                        com += p.pos * p.mass;
                        mass += p.mass;
                    }
                } else{
                    for(int c = m_links[n].firstChild(); c < m_links[n].firstChild() + m_links[n].nChildren(); c++)
                    {
                        min = fmin(min, m_aabbmin[c]);
                        max = fmax(max, m_aabbmax[c]);
                        com += m_openingData[c].com * m_nodeMass[c];
                        mass += m_nodeMass[c];
                    }
                }

                // store the results
                com /= mass;
                m_aabbmin[n] = min;
                m_aabbmax[n] = max;
                m_nodeMass[n] = mass;
                m_openingData[n].com = com;
                m_openingData[n].od2 = calcOpeningDistance2(com, min, max);

            }
        }

        sw.pause();
        std::cout << "Tree update took " << sw.getSeconds() *1000 << "ms" << std::endl;
    }

    void print(int startNode = 0, int count = 0, bool printGeneralInfo = true)
    {
        int endNode = (count == 0) ? m_links.size() : startNode + count;
        std::cout << "printing nodes " << startNode << " to " << endNode << std::endl;
        for(int i = startNode; i < endNode; ++i)
        {
            auto const & l = m_links[i];
            std::cout << "Node " << i << " is leaf: " << l.isLeaf() << " first child: " << l.firstChild()
                      << " num children: " << l.nChildren() << " com: " << m_openingData[i].com << " mass: " << m_nodeMass[i] << std::endl;
        }

        if(printGeneralInfo)
        {
            // check how many children there are

            int tc=0;
            int minp =std::numeric_limits<int>::max();
            int maxp =0;
            for(const auto &l : m_links)
                if(l.isLeaf())
                {
                    tc += l.nChildren();
                    minp = std::min(int(l.nChildren()), minp);
                    maxp = std::max(int(l.nChildren()), maxp);
                }

            int ncnp=0;
            int ncnmin =std::numeric_limits<int>::max();
            int ncnmax =0;
            for(const auto &node : m_criticalNodes)
            {
                ncnp += node.nParticles;
                ncnmin = std::min(node.nParticles, ncnmin);
                ncnmax = std::max(node.nParticles, ncnmax);
            }

            // print some gerneral information
            std::cout << "Number of leafs: " << m_leafs.size() << " with total of " << tc << " particles" << " and min/max/average of " << minp << "/" << maxp << "/" << float(tc) / float(m_leafs.size()) << " particles per leaf" << std::endl;
            std::cout << "Total number of nodes: " << m_links.size() << " in " << m_layerId.size() - 1 << " layers" << std::endl;
            for(int i = 0; i < m_layerId.size()-1; i++)
            {
                std::cout << "\t Layer " << i << " with " << m_layerId[i+1] - m_layerId[i] << " nodes" << std::endl;
            }
            std::cout << "Number of Critical Nodes: " << m_criticalNodes.size() << " with total of " << ncnp << " particles" << " and min/max/average of " << ncnmin << "/" << ncnmax << "/" << float(ncnp) / float(m_criticalNodes.size()) << " particles per critical node" << std::endl;
        }
    }

    template <typename BufferType>
    void computeForces(BufferType& pb)
    {
        mpu::HRStopwatch sw;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_criticalNodes.size()), [&](const auto &range)
        {
            for (auto i = range.begin(); i != range.end(); ++i)
            {
                this->traverseTree(m_criticalNodes[i], pb, &buffer1.local()[0], &buffer2.local()[0], &ilist.local()[0], &nlist.local()[0]);
            }
        });

        sw.pause();
        std::cout << "Computing forces took " << sw.getSeconds() *1000 << "ms" << std::endl;
    }

    template <typename BufferType>
    void computeForcesGPU(BufferType& pb)
    {
        mpu::HRStopwatch sw;

        typename BufferType::deviceType devpb(pb.size());

        CriticalNode* gpuCritNodes;
        f3_t* gpuAabbMin;
        f3_t* gpuAabbMax;
        TreeDownlink* gpuTreeLinks;
        NodeOpeningData* gpuOpeningData;
        f1_t* gpuNodeMass;
        int* gpuGloabalStack;

        cudaMalloc(&gpuCritNodes, m_criticalNodes.size() * sizeof(CriticalNode));
        cudaMalloc(&gpuAabbMin, m_aabbmin.size() * sizeof(f3_t));
        cudaMalloc(&gpuAabbMax, m_aabbmax.size() * sizeof(f3_t));
        cudaMalloc(&gpuTreeLinks, m_links.size() * sizeof(TreeDownlink));
        cudaMalloc(&gpuOpeningData, m_openingData.size() * sizeof(NodeOpeningData));
        cudaMalloc(&gpuNodeMass, m_nodeMass.size() * sizeof(f1_t));
        cudaMalloc(&gpuGloabalStack, stackSizePerBlock * m_criticalNodes.size() * sizeof(int));

        sw.pause();
        std::cout << "GPU Memory allocation " << sw.getSeconds() *1000 << "ms" << std::endl;
        sw.reset();

        devpb = pb;
        cudaMemcpy(gpuCritNodes,m_criticalNodes.data(), m_criticalNodes.size() * sizeof(CriticalNode), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuAabbMin,m_aabbmin.data(), m_aabbmin.size() * sizeof(f3_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuAabbMax,m_aabbmax.data(), m_aabbmax.size() * sizeof(f3_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuTreeLinks,m_links.data(), m_links.size() * sizeof(TreeDownlink), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuOpeningData,m_openingData.data(), m_openingData.size() * sizeof(NodeOpeningData), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuNodeMass,m_nodeMass.data(), m_nodeMass.size() * sizeof(f1_t), cudaMemcpyHostToDevice);
        cudaMemset(gpuGloabalStack,0,stackSizePerBlock * m_criticalNodes.size() * sizeof(int));

        sw.pause();
        std::cout << "Data upload took " << sw.getSeconds() *1000 << "ms" << std::endl;
        sw.reset();

        traverseTreeGPU<<< m_criticalNodes.size(), gpuBlockSize >>>( devpb.getDeviceReference(),
                                                                        gpuCritNodes,
                                                                        gpuAabbMin,
                                                                        gpuAabbMax,
                                                                        gpuTreeLinks,
                                                                        gpuOpeningData,
                                                                        gpuNodeMass,
                                                                        gpuGloabalStack);
        assert_cuda( cudaDeviceSynchronize());

        sw.pause();
        std::cout << "Computing forces took " << sw.getSeconds() *1000 << "ms" << std::endl;
        sw.reset();

        pb = devpb;

        std::cout << "Particle data download took " << sw.getSeconds() *1000 << "ms" << std::endl;

    }

private:
    std::vector<TreeDownlink> m_links; //!< information about the children of a tree node
    std::vector<NodeOpeningData> m_openingData; //!< data needed to check if node should be opened
    std::vector<f1_t> m_nodeMass; //!< total mass of nodes

    std::vector<CriticalNode> m_criticalNodes; //!< nodes that will traverse the tree

    std::vector<f3_t> m_aabbmin; //!< max corner of this nodes axis aligned bounding box
    std::vector<f3_t> m_aabbmax; //!< min corner of this nodes axis aligned bounding box
    std::vector<int> m_uplinks; //!< links to a nodes parent for backwards traversal
    std::vector<int> m_leafs; //!< ids of leaf nodes for backwards traversal
    std::vector<spaceKey> m_nodeKeys; //!< masked morton keys of nodes
    std::vector<int> m_nodeLayer; //!< layer of each node
    std::vector<int> m_layerId; //!< node id where layer x starts

    std::vector<spaceKey> m_mKeys; //!< morten code
    std::vector<unsigned int> m_perm; //!< permutation list for sorting

    // per thread buffer used during traversal only
    tbb::enumerable_thread_specific<std::array<int, stackSizePerThreadCPU>> buffer1;
    tbb::enumerable_thread_specific<std::array<int, stackSizePerThreadCPU>> buffer2;
    tbb::enumerable_thread_specific<std::array<int, interactionListSizeCPU>> ilist;
    tbb::enumerable_thread_specific<std::array<int, nodeListSizeCPU>> nlist;

    template <typename BufferType>
    void calcMortonCodes(const BufferType& pb)
    {
        mpu::HRStopwatch sw;

        // generate morton keys for all particles
        m_mKeys.resize(pb.size());
        const f3_t domainFactor = 1.0_ft / (domainMax - domainMin);
        tbb::parallel_for( tbb::blocked_range<size_t>(0, pb.size()), [&](const auto &range)
        {
            for (auto i = range.begin(); i != range.end(); ++i)
            {
                Particle<POS> p = pb.template loadParticle<POS>(i);
                m_mKeys[i] = calculatePositionKey(p.pos, domainMin, domainFactor);
            }
        });

        sw.pause();
        std::cout << "Morton codes took " << sw.getSeconds() *1000 << "ms" << std::endl;

#ifdef DEBUG_PRINTS
        std::cout << "morton keys generated:\n";
        for(int i =0; i < pb.size(); i++)
        {
            std::bitset<64> x(mKeys[i]);
            std::cout << x << "\n";
        }
        std::cout << std::endl;
#endif
    }

    template <typename BufferType>
    void sortParticles(BufferType& pb)
    {
        mpu::HRStopwatch sw;

        // sort particles by morton key

        // create and sort the permutation vector
        m_perm.resize(pb.size());
        std::iota(m_perm.begin(), m_perm.end(), 0);
        tbb::parallel_sort(m_perm.begin(), m_perm.end(), [keysPtr=m_mKeys.data()](const int idx1, const int idx2) {
            return keysPtr[idx1] < keysPtr[idx2];
        });

        // perform gather to apply the new ordering to the particle buffer
        BufferType sorted(pb.size());
        std::vector<spaceKey> sortedKeys(pb.size());
        tbb::parallel_for( tbb::blocked_range<size_t>(0, pb.size()), [&](const auto &range)
        {
            for (auto i = range.begin(); i != range.end(); ++i)
            {
                const auto pi = pb.loadParticle(m_perm[i]);
                sorted.storeParticle(i,pi);
                sortedKeys[i] = m_mKeys[m_perm[i]];
            }
        });
        pb = std::move(sorted);
        m_mKeys = std::move(sortedKeys);

        sw.pause();
        std::cout << "Sorting codes took " << sw.getSeconds() *1000 << "ms" << std::endl;

#ifdef DEBUG_PRINTS
        std::cout << "morton keys sorted:\n";
    for(int i =0; i < pb.size(); i++)
        std::cout << mKeys[i] << "\n";
    std::cout << std::endl;
#endif
    }

    template <typename BufferType>
    void generateNodes(const BufferType& pb)
    {
        m_links.clear();
        m_layerId.clear();
        m_nodeLayer.clear();
        m_nodeKeys.clear();
        m_criticalNodes.clear();

        mpu::HRStopwatch sw;
        m_links.emplace_back(); // root node
        m_layerId.push_back(0);
        m_nodeLayer.push_back(0);
        m_nodeKeys.push_back(0);

        // temp storage
        std::vector<bool> isInLeaf(pb.size(),false);
        std::vector<bool> isInCritical(pb.size(),false);

        // build the tree layer by layer from top to bottom
        unsigned int layer=1;
        while( layer < 21 ) // TODO: fix layer limit
        {
            spaceKey prevCell;
            bool isInBlock=false; // are we currently inside a block of particles  with the same masked key?

            // node id where next layer starts
            m_layerId.push_back(m_links.size());

            std::vector<int> compactedList;

            spaceKey mask =  ~0llu << (63u-(layer*3u)); // get layer mask
            for(int i = 0; i < pb.size(); i++)
            {
                // for all particles that are not yet in a leaf
                if(!isInLeaf[i])
                {
                    // form regions of particles with same masked keys
                    spaceKey cell = (m_mKeys[i] & mask);
                    if( !isInBlock)
                    {
                        // put i as start of next section
                        prevCell=cell;
                        isInBlock = true;
                        compactedList.push_back(i);
                        m_nodeKeys.push_back(cell);

                    } else if(prevCell != cell)
                    {
                        // end section and start a new one
                        prevCell=cell;
                        compactedList.push_back(i);
                        m_nodeKeys.push_back(cell);
                        compactedList.push_back(i);
                    }

                } else if(isInBlock)
                {
                    // end section
                    compactedList.push_back(i);
                    isInBlock = false;
                }
            }

            // check if work was done this layer
            if(compactedList.empty())
                break;

            if(compactedList.size() % 2 != 0)
            {
                // last section was not ended. That means it is on the end of the buffer. so end it.
                compactedList.push_back(pb.size());
            }

            // go over the compacted list and generate nodes from it
            for(int i=0; i<compactedList.size()-1; i+=2)
            {
                const int id1 = compactedList[i];
                const int id2 = compactedList[i+1];
                const int particlesInNode = id2-id1;
#ifdef DEBUG_PRINTS
                std::cout << "group " << float(i)/2.0f << " particles: " << particlesInNode << std::endl;
#endif
                TreeDownlink link{};

                // do we need to make a critical node?
                if(particlesInNode <= maxCriticalParticles && !isInCritical[id1])
                {
                    m_criticalNodes.push_back(CriticalNode{id1, particlesInNode, static_cast<int>(m_links.size())});
                    for(int j=id1; j<id2; j++)
                        isInCritical[j]=true;
                }

                // is it ok to make a leaf ?
                if( particlesInNode <= maxLeafParticles)
                {
#ifdef DEBUG_PRINTS
                    std::cout << "Leaf" <<std::endl;
#endif
                    link.setData(particlesInNode, id1, true);
                    m_leafs.push_back(m_links.size());
                    for(int j=id1; j<id2; j++)
                        isInLeaf[j]=true;
                }
                m_links.push_back(link);
                m_nodeLayer.push_back(layer);
            }
            layer++;
        }

        assert_critical(layer<=21,"Octree","layer limit reached during node creation")

        // resize all the buffers
        m_uplinks.resize(m_links.size());
        m_aabbmin.resize(m_links.size());
        m_aabbmax.resize(m_links.size());
        m_nodeMass.resize(m_links.size());
        m_openingData.resize(m_links.size());

        // add the end of the node list to the layer list (not needed, this is already done implicitly by the algorithm above)
        // layerId.push_back(links.size());

        sw.pause();
        std::cout << "Generate nodes took " << sw.getSeconds() *1000 << "ms" << std::endl;

        assert_true(m_links.size() == m_nodeLayer.size() && m_links.size() == m_nodeKeys.size(), "Octree", "array sizes do not match after construction")

#ifdef DEBUG_PRINTS
        int tc=0;
        for(const auto &leaf : leafs)
        {
//            std::cout << "node " << leaf << " is leaf with " << links[leaf].nChildren() << " children" << std::endl;
            tc+= links[leaf].nChildren();
        }
        std::cout << "Number of leafs: " << leafs.size() << " with total " << tc << " children" << std::endl;
        std::cout << "Total number of nodes: " << links.size() << " in " << layerId.size()-1 << " layers" << std::endl;

//        std::cout << "printing all nodes after generation (without linking) ..." << std::endl;
//        int nodeid=0;
//        for(const auto &l : links)
//        {
//            std::cout << "Node " << nodeid << " is leaf: " << l.isLeaf() << " first child: " << l.firstChild()
//                      << " num children: " << l.nChildren() << std::endl;
//            nodeid++;
//        }
#endif

    }

    void linkNodes()
    {
        mpu::HRStopwatch sw;

        int lastParent = -1;
        for(int i=1; i < m_links.size(); i++)
        {
            // for each node see on which level the parent is and get some information
            const int parentLayer = m_nodeLayer[i] - 1;
            int parent = (lastParent >= m_layerId[parentLayer]) ? lastParent : m_layerId[parentLayer]; // a parent can have up to 8 children, so start testing with the old parent if it is in the correct layer

            // masking this nodes key with the mask of the parents layer results in a masked key which is equal to the parents
            const spaceKey parentMask =  ~0llu << (63u-(parentLayer*3u));
            const spaceKey thisMaskedKey = (m_nodeKeys[i] & parentMask);

            // get the the key of the current parent candidate
            spaceKey parentMaskedKey = m_nodeKeys[parent];

            // if last nodes parent is not this nodes parent, high chances are it will be close, so do linear search instead of binary
            while(parentMaskedKey != thisMaskedKey)
                parentMaskedKey = m_nodeKeys[++parent];

            assert_true(parent < m_layerId[parentLayer + 1] && parent >= m_layerId[parentLayer], "Octree", "parent assumed in wrong layer")
            assert_true(m_nodeKeys[parent] == thisMaskedKey, "Octree", "Wrong parent assumed during linking.")

            // parent is now the parent, so link it. But first we need to know if we are the first child
            if(lastParent != parent)
                m_links[parent].setFirstChild(i);
                // links start with NChildren == 1
            else
                m_links[parent].setNChildren(m_links[parent].nChildren() + 1);

            // uplink
            m_uplinks[i] = parent;

            // update last parent id
            lastParent = parent;
        }

        sw.pause();
        std::cout << "Link nodes " << sw.getSeconds() *1000 << "ms" << std::endl;

#ifdef DEBUG_PRINTS
        std::cout << "printing all nodes after linking ..." << std::endl;
        int nodeid=0;
        for(const auto &l : links)
        {
            std::cout << "Node " << nodeid << " is leaf: " << l.isLeaf() << " first child: " << l.firstChild()
                      << " num children: " << l.nChildren() << std::endl;
            nodeid++;
        }
#endif

    }

    template <typename BufferType>
    void traverseTree(const CriticalNode& group, BufferType& pb, int *stackWrite, int *stackRead, int *particleInteractionList, int* nodeInteractionList) const
    {
        int *stackEndRead= stackRead;
        int *stackEndWrite= stackWrite;

        int *pilStart= particleInteractionList;
        int *nilStart= nodeInteractionList;

        // add children of root
        for(int child = m_links[0].firstChild(); child < m_links[0].firstChild() + m_links[0].nChildren(); child++)
            *(stackWrite++) = child;

        // swap stacks
        stackRead = stackWrite;
        stackWrite = stackEndRead;
        std::swap(stackEndRead,stackEndWrite);

        const f3_t min = m_aabbmin[group.nodeId];
        const f3_t max = m_aabbmax[group.nodeId];

        // traversal finished when this stack is empty directly after swapping
        while(stackRead != stackEndRead)
        {
            // make this level stack empty
            while(stackRead != stackEndRead)
            {
                int id = *(--stackRead);

                // self interaction is handled later
                if(id == group.nodeId)
                    continue;

                // check if node needs to be opened
                const f1_t r2 = minDistanceSquare(min, max, m_openingData[id].com);
                if(r2 <= m_openingData[id].od2)
                {
                    // it needs to be opened, for leafs interact with all particles, for nodes add them to the next level stack
                    if(m_links[id].isLeaf())
                    {
                        for(int child = m_links[id].firstChild(); child < m_links[id].firstChild() + m_links[id].nChildren(); child++)
                            *(particleInteractionList++) = child;
                    } else
                        for(int child = m_links[id].firstChild(); child < m_links[id].firstChild() + m_links[id].nChildren(); child++)
                            *(stackWrite++) = child;
                } else
                {
                    // no need to open it, so put it into the interaction list
                    *(nodeInteractionList++) = id;
                }
            }
            // swap stacks
            stackRead = stackWrite;
            stackWrite = stackEndRead;
            std::swap(stackEndRead,stackEndWrite);
            assert_critical(stackRead-stackEndRead < stackSizePerThreadCPU, "TREE", "stack overflow during tree travers")
            assert_critical(particleInteractionList-pilStart < interactionListSizeCPU, "TREE", "particle interaction list overflow")
            assert_critical(nodeInteractionList-nilStart < interactionListSizeCPU, "TREE", "node interaction list overflow")
        }

//        std::cout << "particle list size: " << particleInteractionList - pilStart
//                    << " node list size: " << nodeInteractionList - nilStart << std::endl;

        // calculate the interactions
        for(int i = group.firstParticle; i < group.firstParticle + group.nParticles; i++)
        {
            Particle<POS,MASS,ACC> pi = pb.template loadParticle<POS,MASS>(i);

            // with nodes:
            for(int index = 0; index < nodeInteractionList - nilStart; index++)
            {
                int j = nilStart[index];
                f3_t rij = pi.pos - m_openingData[j].com;
                pi.acc += calcInteraction(m_nodeMass[j], dot(rij, rij), rij);
            }

            // with particles in same group:
            for(int j = group.firstParticle; j < group.firstParticle + group.nParticles; j++)
            {
                Particle<POS,MASS> pj = pb.template loadParticle<POS,MASS>(j);
                f3_t rij = pi.pos - pj.pos;
                pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
            }

            // with particles in leafs:
            for(int index = 0; index < particleInteractionList - pilStart; index++)
            {
                Particle<POS,MASS> pj = pb.template loadParticle<POS,MASS>(pilStart[index]);
                f3_t rij = pi.pos - pj.pos;
                pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
            }

            pb.template storeParticle(i,Particle<ACC>(pi));
        }
    }
};

void computeForcesNaive(HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC>& pb)
{
    tbb::parallel_for( tbb::blocked_range<size_t>(0, pb.size()), [&](const auto &range)
    {
        for (auto i = range.begin(); i != range.end(); ++i)
        {
            Particle<POS,MASS,ACC> pi = pb.loadParticle<POS,MASS>(i);
            for(int j = 0; j<pb.size(); j++ )
            {
                Particle<POS,MASS> pj = pb.loadParticle<POS,MASS>(j);
                f3_t rij = pi.pos - pj.pos;
                pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
            }
            pb.storeParticle(i,Particle<ACC>(pi));
        }
    });
}

std::pair<f1_t,f1_t> calcError(const HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC>& pb, const HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC>& reference)
{
    if(pb.size() != reference.size())
        throw std::runtime_error("different particle counts");

    std::vector<f1_t> errors(pb.size());
    for(int i = 0; i < pb.size(); i++)
    {
        auto pi = pb.loadParticle<POS,ACC>(i);
        auto pref = reference.loadParticle<POS,ACC>(i);
#ifdef DEBUG_PRINTS
        std::cout << "Buffer A p" << i << " acc: " << pi.acc << "\tpos: " << pi.pos << std::endl;
        std::cout << "Buffer B p" << i << " acc: " << pref.acc << "\tpos: " << pi.pos << std::endl;
#endif
        errors[i] = length(pi.acc - pref.acc) / length(pref.acc);
    }

    std::sort(std::begin(errors),std::end(errors));
    f1_t median = errors[pb.size()/2];
    f1_t mean = std::accumulate(std::begin(errors),std::end(errors),0.0_ft) / pb.size();


    return std::pair<f1_t,f1_t>(median,mean);
}

f3_t calcTrueCom(HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC>& pb)
{
    f3_t com{};
    f1_t tmass=0;
    for(int i = 0; i < pb.size(); i++)
    {
        auto pi = pb.loadParticle<POS, MASS>(i);
        com += pi.pos * pi.mass;
        tmass += pi.mass;
    }
    return com/tmass;
}

//!< calculate gravity naive on the gpu
struct NaiveGPU
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS>; //!< particle attributes to load from main memory
    using store_type = Particle<ACC>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM>;

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
        pi.acc = {0,0,0};
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        f3_t rij = pi.pos - pj.pos;
        pi.acc += calcInteraction(pj.mass, dot(rij,rij),rij);
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
        return pi;
    }
};

void computeForcesNaiveOnGPU(HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC>& pb)
{
    mpu::HRStopwatch sw;

    DeviceParticleBuffer<DEV_POSM,DEV_VEL,DEV_ACC> devpb(pb.size());

    sw.pause();
    std::cout << "GPU Memory allocation " << sw.getSeconds() *1000 << "ms" << std::endl;
    sw.reset();

    devpb = pb;

    sw.pause();
    std::cout << "Particle data upload took " << sw.getSeconds() *1000 << "ms" << std::endl;
    sw.reset();

    do_for_each_pair_fast<NaiveGPU>(devpb);
    assert_cuda( cudaDeviceSynchronize());

    sw.pause();
    std::cout << "Naive force computation on GPU took " << sw.getSeconds() *1000 << "ms" << std::endl;
    sw.reset();

    pb = devpb;

    std::cout << "Particle data download took " << sw.getSeconds() *1000 << "ms" << std::endl;
}


int main()
{
//    spaceKey k = 7llu << 61u;
//    std::bitset<64> x(k);
//    std::cout << x << "\n";

//    tbb::task_scheduler_init init(1);

    // do some cuda call to remove initialization from timings
    int* blub;
    cudaMalloc(&blub, sizeof(int));

    HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC> pb(1<<15);

    std::default_random_engine rng(161214);
    std::uniform_real_distribution<f1_t > dist(-2,2);

    for(int i =0; i<pb.size(); i++)
    {
        Particle<POS,MASS> p;
        p.mass = 1.0_ft/pb.size();
        p.pos = f3_t{dist(rng),dist(rng),dist(rng)};
        pb.storeParticle(i,p);
    }

#ifdef DEBUG_PRINTS
    for(int i =0; i<pb.size(); i++)
        std::cout << pb.loadParticle(i).pos << "\n";
    std::cout << std::endl;
#endif

    HostTree myTree;

    std::cout << "\n-------------------------------------------\n CONSTRUCTION" << std::endl;

    myTree.construct(pb);
    myTree.update(pb);

    std::cout << "\n-------------------------------------------\n TREE INFO" << std::endl;
    myTree.print(0,1);
    std::cout << "True center of mass: " << calcTrueCom(pb) << std::endl;

    std::cout << "\n-------------------------------------------\n CPU" << std::endl;
    myTree.update(pb);
    myTree.computeForces(pb);

    auto pbRefCpu = pb;
    auto pbRefGPU = pb;
    auto pbGPU = pb;

    mpu::HRStopwatch sw;
    computeForcesNaive(pbRefCpu);
    sw.pause();
    std::cout << "Naive Force computation took " << sw.getSeconds() *1000 << "ms" << std::endl;

    std::cout << "\n-------------------------------------------\n GPU Naive" << std::endl;
    computeForcesNaiveOnGPU(pbRefGPU);

    std::cout << "\n-------------------------------------------\n GPU Tree" << std::endl;
    myTree.computeForcesGPU(pbGPU);

    std::cout << "\n-------------------------------------------\n ERORR" << std::endl;
    std::cout << "CPU Tree" << std::endl;
    auto error = calcError(pb, pbRefCpu);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;

    std::cout << "GPU Naive" << std::endl;
    error = calcError(pbRefCpu, pbRefGPU);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;

    std::cout << "GPU Tree" << std::endl;
    error = calcError(pbRefCpu, pbGPU);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;

    std::cout << "GPU Tree vs CPU Tree" << std::endl;
    error = calcError(pb, pbGPU);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;

    std::cout << "GPU Tree vs GPU Naive" << std::endl;
    error = calcError(pbRefGPU, pbGPU);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;
}