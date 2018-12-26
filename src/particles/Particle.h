/*
 * mpUtils
 * Particle.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_PARTICLE_H
#define MPUTILS_PARTICLE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "particle_tmp_utils.h"
#include "particle_attrib.h"
#include "partice_attributes.h"
//--------------------


//-------------------------------------------------------------------
/**
 * class Particle
 *
 * enables handling and manipulation of the attributes of a single Particle
 *
 * usage:
 * Use the Macro MAKE_PARTICLE_BASE to create different particle attributes. Or use the predefined ones in Particles.h
 * Then pass the name of all the attributes you want to manipulate as template arguments.
 * Example: Particle<POS,MASS> p; p will have a 3D position and a mass.
 * You can use Particle<rPOS,rMASS> ref = p; to create a particle that holds references to the attributes of p.
 *
 */
template <typename... Args>
class Particle : public Args...
{
public:
    static_assert( mpu::conjunction_v< std::is_base_of<particle_attrib,Args>...>,
            "Only use the Particle class with template arguments generated with the macro \"MAKE_PARTICLE_BASE\"! See file ParticleBuffer.h."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<Args...>,particle_base_order >,
            "Use particle Attributes in correct order without duplicates. See particle_base_order in particle_attributes.h.");

    Particle()= default; //!< default construct particle values are undefined

    template <typename... T, std::enable_if_t< mpu::conjunction<mpu::is_list_initable<Args, T&&>...>::value, int> = 0>
    CUDAHOSTDEV
    explicit Particle(T && ... args) : Args(std::forward<T>(args))... {} //!< construct a particle from its attributes

    template <typename... T>
    CUDAHOSTDEV
    Particle(const Particle<T...> &b) : Args(ext_base_cast<Args>(b))... {} //!< construct a particle from another particle with different attributes

    template <typename... T>
    CUDAHOSTDEV Particle<Args...>& operator=(const Particle<T...> &b)
    {
        int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
        (void)t[0]; // silence compiler warning about t being unused
        return *this;
    }
};

//-------------------------------------------------------------------
// create particle in a save way

template<typename Tuple>
struct mp_impl;

template<typename ... TArgs>
struct mp_impl<std::tuple<TArgs...>>
{
    template<typename ...ConstrArgs, std::enable_if_t< (sizeof...(ConstrArgs)>0), int> _null =0>
    static auto make_particle(ConstrArgs && ... args)
    {
        return Particle<TArgs...>(std::forward<ConstrArgs>(args)...);
    }

    template<typename ...ConstrArgs, std::enable_if_t< (sizeof...(ConstrArgs)==0), int> _null =0>
    static auto make_particle(ConstrArgs && ... args)
    {
        return Particle<TArgs...>{};
    }
};

/**
 * @brief Creates a Particle in a save way from a std::tuple of attributes, making sure all attributes are in the correct order Particle Attribute values will be initialised to zero
 * @tparam TupleType the tuple to create the particle from
 * @return the created particle
 */
template <typename TupleType, typename ...ConstrArgs, std::enable_if_t< mpu::is_instantiation_of_v<std::tuple,TupleType> , int> _null =0>
auto make_particle(ConstrArgs && ... args)
{
    return mp_impl< reorderd_t<TupleType,particle_base_order>>::make_particle(std::forward<ConstrArgs>(args)...);
}

/**
 * @brief Creates a Particle in a save way from another particle type (like what you get from concatenating particles), making sure all attributes are in the correct order Particle Attribute values will be initialised to zero
 * @tparam ParticleType the particle to create
 * @return the created particle
 */
template <typename ParticleType, typename ...ConstrArgs, std::enable_if_t< mpu::is_instantiation_of_v<Particle,ParticleType> , int> _null =0>
auto make_particle(ConstrArgs && ... args)
{
    return mp_impl< reorderd_t<particle_to_tuple_t<ParticleType> ,particle_base_order>>::make_particle(std::forward<ConstrArgs>(args)...);
}

/**
 * @brief Creates a Particle in a save way from a list of attributes, making sure all attributes are in the correct order Particle Attribute values will be initialised to zero
 * @tparam TypeArgs particle attributes to sort and use
 * @return the created particle
 */
template <typename ...TypeArgs, typename ...ConstrArgs, std::enable_if_t< (sizeof...(TypeArgs)>1), int> _null =0>
auto make_particle(ConstrArgs && ... args)
{
    return make_particle<std::tuple<TypeArgs...>>(std::forward<ConstrArgs>(args)...);
}

//-------------------------------------------------------------------
// merge two particles

/**
 * @brief Merge two particles. A new particle with all attributes from both input particles is created and values are copied.
 *          If both input particles share an attribute the value from pa is used.
 * @tparam Ta the type of particle A
 * @tparam Tb the type of particle B
 * @param pa the first particle
 * @param pb the second particle
 * @return a new particle with all attributes from pa and pb
 */
template <typename Ta, typename Tb>
auto merge_particles(const Ta& pa, const Tb& pb)
{
    auto p = make_particle< particle_concat_t<Ta,Tb> >(pb);
    p = pa;
    return p;
}

#endif //MPUTILS_PARTICLE_H
