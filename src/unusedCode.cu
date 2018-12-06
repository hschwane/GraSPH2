// save some currently unused functions to integrate back into the code later on

__global__ void generate2DRings(DeviceParticlesType particles)
{

    const float R = 0.38;
    const float r = 0.3;
    const float seperationX = 1;
    const float seperationY = 0;
    const float speed = 0.5;

    const float ringSize = particles.size()/2;
    const float a = M_PI * (R*R-r*r);
    const float ringMass = rho0 * a;
    const float particleMass = ringMass/ringSize;

    // find the starting index
    int startingIndex = (r*r) * ringSize;
    int lastIteration=0;
    while(abs(startingIndex-lastIteration)>5)
    {
        lastIteration = startingIndex;
        startingIndex = ((r/R)*(r/R)) * (ringSize+startingIndex);
    }

    // calculate the particle distance
    f2_t posA;
    f2_t posB;
    float l = R * sqrt(10/(ringSize+startingIndex));
    float theta = 2 * sqrt(M_PIf32*10);
    posA.x = l * cos(theta);
    posA.y = l * sin(theta);
    l = R * sqrt(11/(ringSize+startingIndex));
    theta = 2 * sqrt(M_PIf32*11);
    posB.x = l * cos(theta);
    posB.y = l * sin(theta);
    f1_t spacing = length(posA-posB);
    printf("particle seperation: %f\n ", spacing);

    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
              {
                      thrust::random::default_random_engine rng;
                      rng.discard(i);
                      thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

                      float index;
                      if(i<ringSize)
                      {
                          index = i + startingIndex;
                          pi.pos.x = seperationX/2;
                          pi.pos.y = seperationY/2;
                          pi.vel.x = -speed/2;
                      }
                      else
                      {
                          index = i-ringSize + startingIndex;
                          pi.pos.x = -seperationX/2;
                          pi.pos.y = -seperationY/2;
                          pi.vel.x = speed/2;
                      }

                      l = R * sqrt(index/(ringSize+startingIndex));
                      theta = 2 * sqrt(M_PIf32*index);
                      pi.pos.x += l * cos(theta);
                      pi.pos.y += l * sin(theta);

                      pi.mass = particleMass;
                      pi.density = rho0;
              });
}

__global__ void generateSquares(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
              {
                      float ratio = 0.99f;
                      float spacing = H/3;
                      printf("spacing: %f\n",spacing);
                      int squareSize1 = particles.size() * ratio;
                      int squareSize2 = particles.size() * (1.0f-ratio);
                      int sideres1 = sqrt(float(squareSize1));
                      int sideres2 = sqrt(float(squareSize2));
                      float side1 = (sideres1-1) * spacing;
                      float side2 = (sideres2-1) * spacing;

                      const float a1 = side1*side1;
                      const float a2 = side2*side2;
                      const float squareMass1 = rho0 * a1;
                      const float squareMass2 = rho0 * a2;
                      const float particleMass1 = squareMass1/squareSize1;
                      const float particleMass2 = squareMass2/squareSize2;

                      const float speed = 5;
                      const float seperationX = 1;
                      const float seperationY = 0;

                      thrust::random::default_random_engine rng;
                      rng.discard(i);
                      thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

                      if(i < squareSize1)
                      {
                          pi.pos.x = -side1 / 2 + (i%sideres1) *spacing + dist(rng);
                          pi.pos.y = -side1 / 2 + (i/sideres1) *spacing + dist(rng);
                          pi.pos.x -= seperationX/2;
                          pi.pos.y -= seperationY/2;
                          pi.vel.x = speed * (1-ratio);
                          pi.mass = particleMass1;
                      }
                      else
                      {
                          pi.pos.x = -side2 / 2 + ((i-squareSize1)%sideres2) *spacing + dist(rng);
                          pi.pos.y = -side2 / 2 + ((i-squareSize1)/sideres2) *spacing + dist(rng);
                          pi.pos.x += seperationX/2;
                          pi.pos.y += seperationY/2;
                          pi.vel.x = -speed * ratio;
                          pi.mass = particleMass2;
                      }
                      pi.density = rho0;
              })
}

__global__ void generateRect(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
              {
                      int rows = 150;
                      float offsetY = -0.45;

                      float spacing = H/3;
                      printf("spacing: %f\n",spacing);

                      int coatCols = 2.0f/spacing;
                      int coatSize = coatCols * 3;

                      int size = particles.size() - coatSize;
                      int cols = size / rows;
                      float side1 = (cols-1) * spacing;
                      float side2 = (rows-1) * spacing;
                      const float a = side1*side2;
                      const float mass = rho0 * a;
                      const float particleMass = mass/size;

                      thrust::random::default_random_engine rng;
                      rng.discard(i);
                      thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

                      if(i < coatSize)
                      {
                          pi.pos.x = -1.0f + ((i) % coatCols) * spacing + dist(rng);
                          pi.pos.y = -0.99f + ((i) / coatCols) * spacing + dist(rng);
                      }
                      else
                      {
                          pi.pos.x = -side1 / 2 + ((i-coatSize) % cols) * spacing + dist(rng);
                          pi.pos.y = -side2 / 2 + ((i-coatSize) / cols) * spacing + dist(rng);
                          pi.pos.y += offsetY;
                      }
                      pi.mass = particleMass;
                      pi.density = rho0;
              })
}

__global__ void generate2DHydroNBSystem(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
              {
                      thrust::random::default_random_engine rng;
                      rng.discard(i*particles.size());
                      thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

                      do
                      {
                          pi.pos.x = dist(rng);
                          rng.discard(particles.size());
                          pi.pos.y = dist(rng);
                          rng.discard(particles.size());
                          pi.pos.z = dist(rng);
                          rng.discard(particles.size());
                      }
                      while(length(pi.pos) > 1);

                      pi.mass = mass;
                      pi.density = rho0;

                      pi.vel.x = pi.pos.y * rsqrtf(3);
                      pi.vel.y = - pi.pos.x * rsqrtf(3);
              });
}

__global__ void computeDensity(DeviceParticlesType particles)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM),
                         MPU_COMMA_LIST(POS,MASS,DENSITY),
                         MPU_COMMA_LIST(POS,MASS), MPU_COMMA_LIST(DENSITY),
                         MPU_COMMA_LIST(POS,MASS),
                         {},
                         {
                                 const f3_t rij = pi.pos-pj.pos;
                                 const f1_t r2 = dot(rij,rij);
                                 f1_t r = sqrt(r2);
                                 if(r<=H)
                                 {
                                     pi.density += pj.mass * kernel::Wspline<dimension>(r,H);
                                 }
                         },
                         {})
}

__global__ void window2DBound(DeviceParticlesType particles)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL),
                MPU_COMMA_LIST(POS,VEL),
                MPU_COMMA_LIST(POS,VEL),
                {
                        if(pi.pos.x > 1)
                        {
                            pi.pos.x=1;
                            pi.vel.x -= 1.5*pi.vel.x;
                        }
                        else if(pi.pos.x < -1)
                        {
                            pi.pos.x=-1;
                            pi.vel.x -= 1.5*pi.vel.x;
                        }
                        if(pi.pos.y > 1)
                        {
                            pi.pos.y=1;
                            pi.vel.y -= 1.5*pi.vel.y;
                        }
                        else if(pi.pos.y < -1)
                        {
                            pi.pos.y=-1;
                            pi.vel.y = 0;
                            pi.vel.x = 0;

                        }
                })
}

__global__ void integrate(DeviceParticlesType particles, f1_t dt)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
                MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
                MPU_COMMA_LIST(POS,VEL,DENSITY,DSTRESS),
                {
                        // eqn of motion
                        pi.vel += pi.acc * dt;
                        pi.pos += (pi.vel+0.6f*pi.xvel) * dt;

                        // density
                        pi.density += pi.density_dt * dt;

                        if(pi.density < 0)
                        pi.density = 0;

                        // deviatoric stress
                        pi.dstress += pi.dstress_dt * dt;

                        plasticity(pi.dstress,mohrCoulombYieldStress(tan(friction_angle),eos::murnaghan(pi.density,rho0, BULK, dBULKdP),cohesion));
                })
}

__global__ void generate2DNBSystem(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL),
              {
                      thrust::random::default_random_engine rng;
                      rng.discard(i);
                      thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

                      pi.pos.x = dist(rng);
                      pi.pos.y = dist(rng);
                      pi.pos.z = 0.0f;
                      pi.mass = 1.0f/particles.size();

                      pi.vel = cross(pi.pos,{0.0f,0.0f, 0.75f});
              });
}

__global__ void nbodyForces(DeviceParticlesType particles, f1_t eps2)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, SHARED_POSM, MPU_COMMA_LIST(POS,MASS,VEL,ACC),
                         MPU_COMMA_LIST(POS,MASS,VEL), MPU_COMMA_LIST(ACC), MPU_COMMA_LIST(POS, MASS), {},
                         {
                                 f3_t r = pi.pos - pj.pos;
                                 f1_t distSqr = dot(r, r) + eps2;
                                 f1_t invDist = rsqrt(distSqr);
                                 f1_t invDistCube = invDist * invDist * invDist;
                                 pi.acc -= r * pj.mass * invDistCube;
                         },
                         {
                                 pi.acc -= pi.vel * 0.01;
                         })
}