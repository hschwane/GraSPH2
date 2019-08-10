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
            "Only use the Particle class with template arguments generated with the macro \"MAKE_PARTICLE_BASE\"! See file particle_attributes.h."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<Args...>,particle_base_order >,
            "Use particle Attributes in correct order without duplicates. See particle_base_order in particle_attributes.h.");

    using attributes = std::tuple<Args...>;

    CUDAHOSTDEV static constexpr size_t numAttributes() //!< the number of attributes this particle has
    {
        return std::tuple_size<attributes >::value;
    }

    //!< attribute which attribute T is the derivative of
    template <typename T>
    using isDerivOf_t = typename T::is_derivative_of;

    template <size_t id>
    CUDAHOSTDEV auto getAttribute() //!< get the value of the idth attribute
    {
        return std::tuple_element_t<id,attributes>::getMember();
    }

    template <typename T>
    CUDAHOSTDEV auto getAttribute() //!< get the value of the attribute T
    {
        static_assert( mpu::index_of<T,attributes>::value >= 0, "The attribute you are asking for does not exist.");
        return T::getMember();
    }

    template <size_t id, typename V>
    CUDAHOSTDEV void setAttribute(const V& value) //!< set the value of the idth attribute
    {
        std::tuple_element_t<id,attributes>::setMember(value);
    }

    template <typename T, typename V>
    CUDAHOSTDEV void setAttribute(const V& value) //!< set the value of attribute T
    {
        static_assert(mpu::index_of<T,attributes>::value >= 0, "The attribute you are asking for does not exist.");
        return T::setMember(value);
    }

    template <size_t id>
    CUDAHOSTDEV auto& getAttributeRef() //!< get a reference to the idth attribute
    {
        return std::tuple_element_t<id,attributes>::getMemberRef();
    }

    template <typename T>
    CUDAHOSTDEV auto& getAttributeRef() //!< get a reference to the attribute T
    {
        static_assert(mpu::index_of<T,attributes>::value >= 0, "The attribute you are asking for does not exist.");
        return T::getMemberRef();
    }

    template <size_t id>
    CUDAHOSTDEV const auto& getAttributeRef() const //!< get a const reference to the idth attribute
    {
        return std::tuple_element_t<id,attributes>::getMemberRef();
    }

    template <typename T>
    CUDAHOSTDEV const auto& getAttributeRef() const //!< get a const reference to the attribute T
    {
        static_assert(mpu::index_of<T,attributes>::value >= 0, "The attribute you are asking for does not exist.");
        return T::getMemberRef();
    }

    template <typename F>
    CUDAHOSTDEV void doForEachAttribute(F f) //!< execute functor for each attribute of the particle operator() of f needs to be overloaded on the different datatypes
    {
        int t[] = {0, ((void)(f(Args::getMemberRef())),1)...};
        (void)t[0]; // silence compiler warning about t being unused
    }

    template <typename F>
    CUDAHOSTDEV static void doForEachAttributeType(F f) //!< execute functor for each attribute of the particle operator() of f needs to be templated and work on different particle attributes
    {
        int t[] = {0, ((void)(f.template operator()<Args>()),1)...};
        (void)t[0]; // silence compiler warning about t being unused
    }


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

//!< particle containing all possible attributes (avoid using if at all possible)
using full_particle = mpu::instantiate_from_tuple_t<Particle,particle_base_order >;

//-------------------------------------------------------------------
// create particle in a save way

namespace detail {
    template<typename Tuple>
    struct mp_impl;

    template<typename ... TArgs>
    struct mp_impl<std::tuple<TArgs...>>
    {
        template<typename ...ConstrArgs, std::enable_if_t<(sizeof...(ConstrArgs) > 0), int> _null = 0>
        static auto make_particle(ConstrArgs &&... args)
        {
            return Particle<TArgs...>(std::forward<ConstrArgs>(args)...);
        }

        template<typename ...ConstrArgs, std::enable_if_t<(sizeof...(ConstrArgs) == 0), int> _null = 0>
        static auto make_particle(ConstrArgs &&... args)
        {
            return Particle<TArgs...>{};
        }
    };
}

/**
 * @brief Creates a Particle in a save way from a std::tuple of attributes, making sure all attributes are in the correct order Particle Attribute values will be initialised to zero
 * @tparam TupleType the tuple to create the particle from
 * @return the created particle
 */
template <typename TupleType, typename ...ConstrArgs, std::enable_if_t< mpu::is_instantiation_of_v<std::tuple,TupleType> , int> _null =0>
auto make_particle(ConstrArgs && ... args)
{
    return detail::mp_impl< reorderd_t<TupleType,particle_base_order>>::make_particle(std::forward<ConstrArgs>(args)...);
}

/**
 * @brief Creates a Particle in a save way from another particle type (like what you get from concatenating particles), making sure all attributes are in the correct order Particle Attribute values will be initialised to zero
 * @tparam ParticleType the particle to create
 * @return the created particle
 */
template <typename ParticleType, typename ...ConstrArgs, std::enable_if_t< mpu::is_instantiation_of_v<Particle,ParticleType> , int> _null =0>
auto make_particle(ConstrArgs && ... args)
{
    return detail::mp_impl< reorderd_t<particle_to_tuple_t<ParticleType> ,particle_base_order>>::make_particle(std::forward<ConstrArgs>(args)...);
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

/**
 * @brief The type of particle generated when calling make particle with the template arguments Args
 */
template <typename ...Args>
using make_particle_t = decltype(make_particle<Args...>());

//-------------------------------------------------------------------
// merge multiple particles

/**
 * @brief Merge multiple particles. A new particle with all attributes from all input particles is created and values are copied.
 *          If input particles share an attribute the value from particle Pa or the last particle with the attribute in question is used.
 * @param pa the first particle
 * @param pb the second particle
 * @return a new particle with all attributes from pa and pb
 */
template <typename PTa, typename ...PTs>
auto merge_particles(const PTa& pa,const PTs& ... ps)
{
    auto p = make_particle< particle_concat_t<PTa, PTs...> >();

    int t[] = {0, ((void)(p=ps),1)...};
    (void)t[0]; // silence compiler warning about t being unused
    p = pa;

    return p;
}

template <typename ... Args>
using merge_particles_t = decltype(merge_particles<Args...>(Args()...));

#endif //MPUTILS_PARTICLE_H
