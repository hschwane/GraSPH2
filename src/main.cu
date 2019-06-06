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
constexpr int maxLeafParticles = 16;
constexpr int maxCriticalParticles = 256;
constexpr int stackSizePerThread = 4096;
constexpr int interactionListSize = 40000;
constexpr int nodeListSize = 1000;
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
    CUDAHOSTDEV void setNChildren(unsigned int nchildren) { m_data = (m_data & ~(numMask)) | ((nchildren-1) << numShift);} //!< if you put in a number outside of [1,16] the universe will explode!
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

struct CriticalNode
{
    int firstParticle;
    int nParticles;
    int nodeId;
};

f1_t calcOpeningDistance2(const f3_t& com, const f3_t& min, const f3_t& max)
{
    f3_t l3d = fabs(max - min);
    f1_t l = fmax(fmax(l3d.x,l3d.y),l3d.z);
    f3_t cogeo = min + (l3d*0.5_ft);
    f1_t delta = length(com-cogeo);
    f1_t od = l / theta + delta;
    f1_t od2 = od*od;
    return (std::isnan(od2) ? std::numeric_limits<f1_t>::infinity() : od2);
}

f3_t calcInteraction( f1_t massj, f1_t r2, f3_t rij)
{
    // gravity
    const f1_t distSqr = r2 + eps2;
    const f1_t invDist = rsqrt(distSqr);
    const f1_t invDistCube = invDist * invDist * invDist;
    return -rij * massj * invDistCube;
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

        for(int i = layerId.size()-2; i >= 0; i-- )
        {
            // for each layer loop over all nodes in it
            int layerStart = layerId[i];
            int layerEnd = layerId[i+1];
            for(int n = layerStart; n < layerEnd; n++)
            {
                // for each node go over all leafs to calculate com, total mass and bounding box
                f3_t min{std::numeric_limits<f1_t>::infinity(),std::numeric_limits<f1_t>::infinity(),std::numeric_limits<f1_t>::infinity()};
                f3_t max{0,0,0};
                f3_t com{0,0,0};
                f1_t mass=0;

                if(links[n].isLeaf())
                {
                    for(int c = links[n].firstChild(); c < links[n].firstChild() + links[n].nChildren(); c++)
                    {
                        auto p = pb.template loadParticle<POS,MASS>(c);
                        min = fmin(min, p.pos);
                        max = fmax(max, p.pos);
                        com += p.pos * p.mass;
                        mass += p.mass;
                    }
                } else{
                    for(int c = links[n].firstChild(); c < links[n].firstChild() + links[n].nChildren(); c++)
                    {
                        min = fmin(min, aabbmin[c]);
                        max = fmax(max, aabbmax[c]);
                        com += openingData[c].com * nodeMass[c];
                        mass += nodeMass[c];
                    }
                }

                // store the results
                com /= mass;
                aabbmin[n] = min;
                aabbmax[n] = max;
                nodeMass[n] = mass;
                openingData[n].com = com;
                openingData[n].od2 = calcOpeningDistance2(com, min, max);

            }
        }

        sw.pause();
        std::cout << "Tree update took " << sw.getSeconds() *1000 << "ms" << std::endl;
    }

    void print(int startNode = 0, int count = 0, bool printGeneralInfo = true)
    {
        int tc=0;
        int endNode = (count == 0) ? links.size() : startNode + count;
        std::cout << "printing nodes " << startNode << " to " << endNode << std::endl;
        for(int i = startNode; i < endNode; ++i)
        {
            auto const & l = links[i];
            std::cout << "Node " << i << " is leaf: " << l.isLeaf() << " first child: " << l.firstChild()
                      << " num children: " << l.nChildren() << " com: " << openingData[i].com << " mass: " << nodeMass[i] << std::endl;
        }

        if(printGeneralInfo)
        {
            // check how many children there are
            for(const auto &l : links)
                if(l.isLeaf())
                    tc+=l.nChildren();

            int ncnp=0;
            for(const auto &node : criticalNodes)
            {
                ncnp += node.nParticles;
            }

            // print some gerneral information
            std::cout << "Number of leafs: " << leafs.size() << " with total " << tc << " children" << std::endl;
            std::cout << "Total number of nodes: " << links.size() << " in " << layerId.size()-1 << " layers" << std::endl;
            std::cout << "Number of Critical Nodes: " << criticalNodes.size() << " with total of " << ncnp << " particles" << std::endl;
        }
    }

    template <typename BufferType>
    void computeForces(BufferType& pb)
    {
        mpu::HRStopwatch sw;

        tbb::parallel_for( tbb::blocked_range<size_t>(0, criticalNodes.size()), [&](const auto &range)
        {
            std::array<int, stackSizePerThread> buffer1;
            std::array<int, stackSizePerThread> buffer2;
            std::array<int, interactionListSize> ilist;
            std::array<int, nodeListSize> nlist;

            for (auto i = range.begin(); i != range.end(); ++i)
            {
                this->traverseTree(criticalNodes[i],pb, &buffer1[0], &buffer2[0], &ilist[0], &nlist[0]);
            }
        });

        sw.pause();
        std::cout << "Computing forces took " << sw.getSeconds() *1000 << "ms" << std::endl;
    }

private:
    std::vector<TreeDownlink> links; //!< information about the children of a tree node
    std::vector<NodeOpeningData> openingData; //!< data needed to check if node should be opened
    std::vector<f1_t> nodeMass; //!< total mass of nodes

    std::vector<CriticalNode> criticalNodes; //!< nodes that will traverse the tree

    std::vector<f3_t> aabbmin; //!< max corner of this nodes axis aligned bounding box
    std::vector<f3_t> aabbmax; //!< min corner of this nodes axis aligned bounding box
    std::vector<int> uplinks; //!< links to a nodes parent for backwards traversal
    std::vector<int> leafs; //!< ids of leaf nodes for backwards traversal
    std::vector<spaceKey> nodeKeys; //!< masked morton keys of nodes
    std::vector<int> nodeLayer; //!< layer of each node
    std::vector<int> layerId; //!< node id where layer x starts

    std::vector<spaceKey> mKeys; //!< morten code
    std::vector<unsigned int> perm; //!< permutation list for sorting

    template <typename BufferType>
    void calcMortonCodes(const BufferType& pb)
    {
        mpu::HRStopwatch sw;

        // generate morton keys for all particles
        mKeys.resize(pb.size());
        const f3_t domainFactor = 1.0_ft / (domainMax - domainMin);
        tbb::parallel_for( tbb::blocked_range<size_t>(0, pb.size()), [&](const auto &range)
        {
            for (auto i = range.begin(); i != range.end(); ++i)
            {
                Particle<POS> p = pb.template loadParticle<POS>(i);
                mKeys[i] = calculatePositionKey(p.pos,domainMin,domainFactor);
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
        perm.resize(pb.size());
        std::iota(perm.begin(),perm.end(),0);
        tbb::parallel_sort(perm.begin(),perm.end(), [keysPtr=mKeys.data()](const int idx1, const int idx2) {
            return keysPtr[idx1] < keysPtr[idx2];
        });

        // perform gather to apply the new ordering to the particle buffer
        BufferType sorted(pb.size());
        std::vector<spaceKey> sortedKeys(pb.size());
        tbb::parallel_for( tbb::blocked_range<size_t>(0, pb.size()), [&](const auto &range)
        {
            for (auto i = range.begin(); i != range.end(); ++i)
            {
                const auto pi = pb.loadParticle(perm[i]);
                sorted.storeParticle(i,pi);
                sortedKeys[i] = mKeys[perm[i]];
            }
        });
        pb = std::move(sorted);
        mKeys = std::move(sortedKeys);

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
        links.clear();
        layerId.clear();
        nodeLayer.clear();
        nodeKeys.clear();
        criticalNodes.clear();

        mpu::HRStopwatch sw;
        links.emplace_back(); // root node
        layerId.push_back(0);
        nodeLayer.push_back(0);
        nodeKeys.push_back(0);

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
            layerId.push_back(links.size());

            std::vector<int> compactedList;

            spaceKey mask =  ~0llu << (63u-(layer*3u)); // get layer mask
            for(int i = 0; i < pb.size(); i++)
            {
                // for all particles that are not yet in a leaf
                if(!isInLeaf[i])
                {
                    // form regions of particles with same masked keys
                    spaceKey cell = (mKeys[i] & mask);
                    if( !isInBlock)
                    {
                        // put i as start of next section
                        prevCell=cell;
                        isInBlock = true;
                        compactedList.push_back(i);
                        nodeKeys.push_back(cell);

                    } else if(prevCell != cell)
                    {
                        // end section and start a new one
                        prevCell=cell;
                        compactedList.push_back(i);
                        nodeKeys.push_back(cell);
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
                    criticalNodes.push_back(CriticalNode{id1,particlesInNode, static_cast<int>(links.size())});
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
                    leafs.push_back(links.size());
                    for(int j=id1; j<id2; j++)
                        isInLeaf[j]=true;
                }
                links.push_back(link);
                nodeLayer.push_back(layer);
            }
            layer++;
        }

        assert_critical(layer<=21,"Octree","layer limit reached during node creation")

        // resize all the buffers
        uplinks.resize(links.size());
        aabbmin.resize(links.size());
        aabbmax.resize(links.size());
        nodeMass.resize(links.size());
        openingData.resize(links.size());

        // add the end of the node list to the layer list (not needed, this is already done implicitly by the algorithm above)
        // layerId.push_back(links.size());

        sw.pause();
        std::cout << "Generate nodes took " << sw.getSeconds() *1000 << "ms" << std::endl;

        assert_true(links.size() == nodeLayer.size() && links.size() == nodeKeys.size(),"Octree","array sizes do not match after construction")

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
        for(int i=1; i < links.size(); i++)
        {
            // for each node see on which level the parent is and get some information
            const int parentLayer = nodeLayer[i]-1;
            int parent = (lastParent >= layerId[parentLayer]) ? lastParent : layerId[parentLayer]; // a parent can have up to 8 children, so start testing with the old parent if it is in the correct layer

            // masking this nodes key with the mask of the parents layer results in a masked key which is equal to the parents
            const spaceKey parentMask =  ~0llu << (63u-(parentLayer*3u));
            const spaceKey thisMaskedKey = (nodeKeys[i] & parentMask);

            // get the the key of the current parent candidate
            spaceKey parentMaskedKey = nodeKeys[parent];

            // if last nodes parent is not this nodes parent, high chances are it will be close, so do linear search instead of binary
            while(parentMaskedKey != thisMaskedKey)
                parentMaskedKey = nodeKeys[++parent];

            assert_true( parent < layerId[parentLayer+1] && parent >= layerId[parentLayer], "Octree", "parent assumed in wrong layer")
            assert_true( nodeKeys[parent] == thisMaskedKey, "Octree", "Wrong parent assumed during linking.")

            // parent is now the parent, so link it. But first we need to know if we are the first child
            if(lastParent != parent)
                links[parent].setFirstChild(i);
                // links start with NChildren == 1
            else
                links[parent].setNChildren( links[parent].nChildren()+1);

            // uplink
            uplinks[i] = parent;

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
        for(int child = links[0].firstChild(); child < links[0].firstChild() + links[0].nChildren(); child++)
            *(stackWrite++) = child;

        // swap stacks
        stackRead = stackWrite;
        stackWrite = stackEndRead;
        std::swap(stackEndRead,stackEndWrite);

        f3_t com = openingData[group.nodeId].com;

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
                const f3_t rij = com - openingData[id].com;
                const f1_t r2 = dot(rij,rij);
                if(r2 < openingData[id].od2)
                {
                    // it needs to be opened, for leafs interact with all particles, for nodes add them to the next level stack
                    if(links[id].isLeaf())
                    {
                        for(int child = links[id].firstChild(); child < links[id].firstChild() + links[id].nChildren(); child++)
                            *(particleInteractionList++) = child;
                    } else
                        for(int child = links[id].firstChild(); child < links[id].firstChild() + links[id].nChildren(); child++)
                            *(stackWrite++) = child;
                } else
                {
                    // no need to open it, so interact with it already
                    *(nodeInteractionList++) = id;
                }
            }
            // swap stacks
            stackRead = stackWrite;
            stackWrite = stackEndRead;
            std::swap(stackEndRead,stackEndWrite);
            assert_true(stackRead-stackEndRead < stackSizePerThread,"TREE","stack overflow during tree travers")
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
                f3_t rij = pi.pos - openingData[j].com;
                pi.acc += calcInteraction(nodeMass[j], dot(rij,rij),rij);
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


int main()
{
//    spaceKey k = 7llu << 61u;
//    std::bitset<64> x(k);
//    std::cout << x << "\n";

//    tbb::task_scheduler_init init(1);

    HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC> pb(50000);

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
    myTree.construct(pb);
    myTree.update(pb);

    myTree.computeForces(pb);

    auto pbRef = pb;

    mpu::HRStopwatch sw;
    computeForcesNaive(pbRef);
    std::cout << "Naive Force computation took " << sw.getSeconds() *1000 << "ms" << std::endl;
//
//    std::cout << "treecode used " << interactions << " interactions. Naive: " << pb.size()*pb.size() << ". Saved: " << pb.size()*pb.size() - interactions << " relative: " << 1.0_ft*(pb.size()*pb.size() - interactions) / (pb.size()*pb.size())  << std::endl;
//    std::cout << "average leafs opened: " << 1.0_ft*averageLeafs / pb.size() << std::endl;
//
    auto error = calcError(pb,pbRef);
    std::cout << "Median error: " << error.first << std::endl;
    std::cout << "Mean error: " << error.second << std::endl;

    myTree.print(0,1);

}



///**
// * @brief first pass of derivative computation
// */
//struct cdA
//{
//    // define particle attributes to use
//    using load_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
//    using store_type = Particle<BALSARA,DENSITY_DT,DSTRESS_DT>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//    using pj_type = Particle<POS,MASS,VEL,DENSITY>; //!< the particle attributes to load from main memory of all the interaction partners j
//    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
//    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_DENSITY>;
//
//    // setup some variables we need before during and after the pair interactions
//    m3_t edot{0}; // strain rate tensor (edot)
//    m3_t rdot{0}; // rotation rate tensor
//    f1_t divv{0}; // velocity divergence
//
//    //!< This function is executed for each particle before the interactions are computed.
//    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
//    {
//    }
//
//    //!< This function will be called for each pair of particles.
//    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
//    {
//        const f3_t rij = pi.pos-pj.pos;
//        const f1_t r2 = dot(rij,rij);
//        if(r2 <= H2 && r2>0)
//        {
//            // get the kernel gradient
//            f1_t r = sqrt(r2);
//            const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
//            const f3_t gradw = (dw / r) * rij;
//
//            // strain rate tensor (edot) and rotation rate tensor (rdot)
//            const f3_t vij = pi.vel - pj.vel;
//            addStrainRateAndRotationRate(edot,rdot,pj.mass,pj.density,vij,gradw);
//            divv -= (pj.mass / pj.density) * dot(vij, gradw);
//        }
//    }
//
//    //!< This function will be called for particle i after the interactions with the other particles are computed.
//    CUDAHOSTDEV store_type do_after(pi_type& pi)
//    {
//        // deviatoric stress time derivative
//        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
//
//        // density time derivative
//        pi.density_dt = -pi.density * divv;
//
//#if defined(BALSARA_SWITCH)
//        // get curl from edot and compute the balsara switch value
//        const f3_t curlv = f3_t{-2*rdot[1][2], -2*rdot[2][0], -2*rdot[0][1] };
//        pi.balsara = balsaraSwitch(divv, curlv, SOUNDSPEED, H);
//#endif
//
//        return pi;
//    }
//};
//
///**
// * @brief second pass of derivative computation
// */
//struct cdB
//{
//    // define particle attributes to use
//    using load_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
//    using store_type = Particle<ACC,XVEL>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//    using pj_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
//    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
//    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_BALSARA,SHARED_DENSITY,SHARED_DSTRESS>;
//
//    // setup some variables we need before during and after the pair interactions
//#if defined(ENABLE_SPH)
//    m3_t sigOverRho_i; // stress over density square used for acceleration
//    #if defined(ARTIFICIAL_STRESS)
//        m3_t arts_i; // artificial stress from i
//    #endif
//#endif
//
//    //!< This function is executed for each particle before the interactions are computed.
//    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
//    {
//#if defined(ENABLE_SPH)
//        // build stress tensor for particle i using deviatoric stress and pressure
//        m3_t sigma_i = pi.dstress;
//        const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
//        sigma_i[0][0] -= pres_i;
//        sigma_i[1][1] -= pres_i;
//        sigma_i[2][2] -= pres_i;
//
//        sigOverRho_i = sigma_i / (pi.density*pi.density);
//
//    #if defined(ARTIFICIAL_STRESS)
//        // artificial stress from i
//        m3_t arts_i = artificialStress(mateps, sigOverRho_i);
//    #endif
//#endif
//    }
//
//    //!< This function will be called for each pair of particles.
//    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
//    {
//        const f3_t rij = pi.pos-pj.pos;
//        const f1_t r2 = dot(rij,rij);
//        if(r2>0)
//        {
//
//#if defined(ENABLE_SELF_GRAVITY)
//            // gravity
//            const f1_t distSqr = r2 + H2;
//            const f1_t invDist = rsqrt(distSqr);
//            const f1_t invDistCube = invDist * invDist * invDist;
//            pi.acc -= rij * pj.mass * invDistCube;
//#endif
//
//#if defined(ENABLE_SPH)
//            if(r2 <= H2)
//            {
//                // get the kernel gradient
//                f1_t r = sqrt(r2);
//                const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
//                const f3_t gradw = (dw / r) * rij;
//
//                // stress and pressure of j
//                m3_t sigma_j = pj.dstress;
//                const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
//                sigma_j[0][0] -= pres_j;
//                sigma_j[1][1] -= pres_j;
//                sigma_j[2][2] -= pres_j;
//
//                m3_t sigOverRho_j = sigma_j / (pj.density * pj.density);
//
//                // stress from the interaction
//                m3_t stress = sigOverRho_i + sigOverRho_j;
//
//                const f3_t vij = pi.vel - pj.vel;
//    #if defined(ARTIFICIAL_STRESS)
//                // artificial stress
//                const f1_t f = pow(kernel::Wspline(r, H, W_prefactor) / kernel::Wspline(normalsep, H, W_prefactor) , matexp;
//                stress += f*(arts_i + artificialStress(mateps, sigOverRho_j));
//    #endif
//
//                // acceleration from stress
//                pi.acc += pj.mass * (stress * gradw);
//
//    #if defined(ARTIFICIAL_VISCOSITY)
//                // acceleration from artificial viscosity
//                pi.acc -= pj.mass *
//                          artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, SOUNDSPEED, SOUNDSPEED
//        #if defined(BALSARA_SWITCH)
//                                  , pi.balsara, pj.balsara
//        #endif
//                          ) * gradw;
//    #endif
//
//    #if defined(XSPH)
//                // xsph
//                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
//    #endif
//            }
//#endif // ENABLE_SPH
//        }
//    }
//
//    //!< This function will be called for particle i after the interactions with the other particles are computed.
//    CUDAHOSTDEV store_type do_after(pi_type& pi)
//    {
//#if defined(CLOHESSY_WILTSHIRE)
//        pi.acc.x += 3*cw_n*cw_n * pi.pos.x + 2*cw_n* pi.vel.y;
//        pi.acc.y += -2*cw_n * pi.vel.x;
//        pi.acc.z += -cw_n*cw_n * pi.pos.z;
//#endif
//        return pi;
//    }
//};
//
//template <typename pbT>
//void computeDerivatives(pbT& particleBuffer)
//{
//#if defined(ENABLE_SPH)
//    do_for_each_pair_fast<cdA>(particleBuffer);
//#endif
//    do_for_each_pair_fast<cdB>(particleBuffer);
//}
//
///**
// * @brief perform leapfrog integration on the particles also performs the plasticity calculations
// * @param particles the device copy to the particle buffer that stores the particles
// * @param dt the timestep for the integration
// * @param not_first_step set false for the first integration step of the simulation
// * @param tanfr tangens of the internal friction angle
// */
//struct integrateLeapfrog
//{
//    using load_type = Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>; //!< particle attributes to load from main memory
//    using store_type = Particle<POS,VEL,DENSITY,DSTRESS>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//
//    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
//    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
//    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, f1_t dt, bool not_first_step)
//    {
//        //   calculate velocity a_t
//        p.vel = p.vel + p.acc * (dt * 0.5_ft);
//
//        // we could now change delta t here
//
//        // calculate velocity a_t+1/2
//        p.vel = p.vel + p.acc * (dt * 0.5_ft) * not_first_step;
//
//        // calculate position r_t+1
//#if defined(XSPH) && defined(ENABLE_SPH)
//        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
//#else
//        p.pos = p.pos + p.vel * dt;
//#endif
//
//#if defined(ENABLE_SPH)
//        // density
//        p.density = p.density + p.density_dt * dt;
//        if(p.density < 0.0_ft)
//            p.density = 0.0_ft;
//
//        // deviatoric stress
//        p.dstress += p.dstress_dt * dt;
//
//    #if defined(PLASTICITY_MC)
//        plasticity(p.dstress, mohrCoulombYieldStress( tanfr,eos::murnaghan(p.density,rho0, BULK, dBULKdP),cohesion));
//    #elif defined(PLASTICITY_MIESE)
//        plasticity(p.dstress,Y);
//    #endif
//
//#endif
//        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
//    }
//};
//
///**
// * @brief The main function of the simulation. Sets up the initial conditions and frontend and then manages running the simulation.
// *
// */
//
//
//int main()
//{
//    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());
//
//    std::string buildType;
//#if defined(NDEBUG)
//    buildType = "Release";
//#else
//    buildType = "Debug";
//#endif
//
//#if defined(STORE_RESULTS)
//    // set up file saving engine
//    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
//    // setup log output file
//    myLog.addSinks(mpu::FileSink( std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_log.txt"));
//    // collect all settings and print them into a file
//    {
//        mpu::Resource headlessSettings = LOAD_RESOURCE(HeadlessSettings);
//        mpu::Resource precisionSettings = LOAD_RESOURCE(PrecisionSettings);
//        mpu::Resource settings = LOAD_RESOURCE(Settings);
//        std::ofstream settingsOutput(std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_settings.txt");
//        settingsOutput << "//////////////////////////\n// headlessSettigns.h \n//////////////////////////\n\n"
//                        << std::string(headlessSettings.data(), headlessSettings.size())
//                        << "\n\n\n//////////////////////////\n// precisionSettings.h \n//////////////////////////\n\n"
//                        << std::string(precisionSettings.data(), precisionSettings.size())
//                        << "\n\n\n//////////////////////////\n// settigns.h \n//////////////////////////\n\n"
//                        << std::string(settings.data(), settings.size());
//    }
//#endif
//
//    myLog.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
//    logINFO("GraSPH2") << "Welcome to GraSPH2!";
//#if defined(SINGLE_PRECISION)
//    logINFO("GraSPH2") << "Running in single precision mode.";
//#elif defined(DOUBLE_PRECISION)
//    logINFO("GraSPH2") << "Running in double precision mode.";
//#endif
//#if defined(USING_CUDA_FAST_MATH)
//    logWARNING("GraSPH2") << "Unsafe math optimizations enabled in CUDA code.";
//#endif
//    assert_cuda(cudaSetDevice(0));
//
//    // print some important settings to the console
//    myLog.print(mpu::LogLvl::INFO) << "\nSettings for this run:\n========================\n"
//                        << "Integration:"
//                        << "Leapfrog"
//                        << "Timestep: constant, " << timestep << "\n"
//                        << "Initial Conditions:\n"
//                #if defined(READ_FROM_FILE)
//                        << "Data is read from: " << FILENAME << "\n"
//                #elif defined(ROTATING_UNIFORM_SPHERE)
//                        << "Using a random uniform sphere with radius " << spawn_radius << "\n"
//                        << "Total mass: " << tmass << "\n"
//                        << "Number of particles: " << particle_count << "\n"
//                        << "additional angular velocity: " << angVel << "\n"
//                #elif defined(ROTATING_PLUMMER_SPHERE)
//                        << "Using a Plummer distribution with core radius " << plummer_radius << " and cutoff " << plummer_cutoff << "\n"
//                        << "Total mass: " << tmass << "\n"
//                        << "Number of particles: " << particle_count << "\n"
//                        << "additional angular velocity: " << angVel << "\n"
//                #endif
//                        << "Compressed radius set to " << compressesd_radius << "\n"
//                        << "resulting in particle radius of " << pradius << "\n"
//                        << "and smoothing length " << H << "\n"
//                        << "Material Settings:\n"
//                        << "material density: " << rho0 << "\n"
//                        << "speed of sound: " << SOUNDSPEED << "\n"
//                        << "bulk-modulus: " << BULK << "\n"
//                        << "shear-modulus: " << shear << "\n"
//                        << "Environment Settings:\n"
//                #if defined(CLOHESSY_WILTSHIRE)
//                        << "Clohessy-Wiltshire enabled with n = " << cw_n << "\n";
//                #else
//                        << "Clohessy-Wiltshire disabled" << "\n"
//                #endif
//                        ;
//
//    // set up frontend
//    fnd::initializeFrontend();
//    bool simShouldRun = false;
//    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});
//
//    // generate some particles depending on options in the settings file
//    InitGenerator<HostParticlesType> generator;
//
//#if defined(READ_FROM_FILE)
//    generator.addParticles(ps::TextFile<particleToRead>(FILENAME,SEPERATOR));
//#elif defined(ROTATING_UNIFORM_SPHERE)
//    generator.addParticles( ps::UniformSphere(particle_count,spawn_radius,tmass,rho0).addAngularVelocity(angVel), true,true );
//#elif defined(ROTATING_PLUMMER_SPHERE)
//    generator.addParticles( ps::PlummerSphere(particle_count,plummer_radius,plummer_cutoff,tmass,rho0).addAngularVelocity(angVel), true, true);
//#endif
//
//    auto hpb = generator.generate();
//
//    // create cuda buffer
//    DeviceParticlesType pb(hpb.size());
//#if defined(FRONTEND_OPENGL)
//    fnd::setParticleSize(pradius);
//    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
//    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
//    pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
//    pb.mapGraphicsResource();
//#endif
//
//    // upload particles
//    pb = hpb;
//
//#if defined(STORE_RESULTS)
//    // print timestep 0
//    storage.printToFile(pb,0);
//    f1_t timeSinceStore=timestep;
//#endif
//
//    // start simulating
//    computeDerivatives(pb);
//    do_for_each<integrateLeapfrog>(pb,timestep,false);
//
//    double simulatedTime=timestep;
//#if defined(READ_FROM_FILE)
//    simulatedTime += startTime;
//#endif
//
//    pb.unmapGraphicsResource(); // used for frontend stuff
//    while(fnd::handleFrontend(simulatedTime))
//    {
//        if(simShouldRun)
//        {
//            pb.mapGraphicsResource(); // used for frontend stuff
//
//            // run simulation
//            computeDerivatives(pb);
//            do_for_each<integrateLeapfrog>(pb,timestep,true);
//
//            simulatedTime += timestep;
//
//#if defined(STORE_RESULTS)
//            timeSinceStore += timestep;
//            if( timeSinceStore >= store_intervall)
//            {
//                storage.printToFile(pb,simulatedTime);
//                timeSinceStore-=store_intervall;
//            }
//#endif
//
//            pb.unmapGraphicsResource(); // used for frontend stuff
//        }
//    }
//
//    return 0;
//}
//
