########################################################################################################################
#                                                       DEV                                                            #
########################################################################################################################
FROM maven:3.6.1-jdk-8 as DEV

##########################
#  install dependencies  #
##########################
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl=7.52.1-5+deb9u9 \
       g++=4:6.3.0-4 \
       make=4.1-9.1 \
       unzip=6.0-21+deb9u1 \
    && rm -rf /var/lib/apt/lists/*

#######################
#  install new cmake  #
#######################
RUN curl -fsSL --insecure -o /tmp/cmake.tar.gz https://cmake.org/files/v3.13/cmake-3.13.4.tar.gz \
    && tar -xzf /tmp/cmake.tar.gz -C /tmp \
    && rm -rf /tmp/cmake.tar.gz  \
    && mv /tmp/cmake-* /tmp/cmake \
    && cd /tmp/cmake \
    && ./bootstrap \
    && make -j4 \
    && make install \
    && rm -rf /tmp/cmake

#######################
#  download libtorch  #
#######################
WORKDIR /opt
ENV LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0%2Bcpu.zip
RUN curl -fsSL --insecure -o libtorch.zip  $LIBTORCH_URL \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

ENV TORCH_HOME=/opt/libtorch

#####################
#  Install PyTorch  #
#####################
WORKDIR /tmp
RUN curl -fsSL --insecure -o anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh \
    && /bin/bash anaconda.sh -b -p /opt/conda \
    && rm anaconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> "$HOME"/.bashrc \
    && echo "conda activate base" >> "$HOME"/.bashrc

ENV PATH /opt/conda/bin/:$PATH

RUN /opt/conda/bin/conda install -yq \
      -c pytorch \
      pytorch=1.2.0 \
      torchvision=0.4.0 \
      cpuonly=1.0  \
    && /opt/conda/bin/conda clean -yq --all


########################################################################################################################
#                                                     JAVA BUILDER                                                     #
########################################################################################################################
FROM DEV as JAVA_BUILDER


WORKDIR /app

COPY ./java/pom.xml /app

RUN mvn -e -B dependency:resolve dependency:resolve-plugins

COPY ./java /app

RUN mvn -e -B -Dmaven.test.skip=true package
########################################################################################################################
#                                                     CPP BUILDER                                                      #
########################################################################################################################
FROM DEV as CPP_BUILDER

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
       zip=3.0-11+b1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./cpp ./

RUN ./build.sh \
    && cp ./out/*.so "$TORCH_HOME"/lib \
    && cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$TORCH_HOME"/lib \
    && ln -s "$TORCH_HOME"/lib libtorch \
    && zip -qr /angel_libtorch.zip libtorch

########################################################################################################################
#                                                       Artifacts                                                      #
########################################################################################################################
FROM alpine:3.10 as ARTIFACTS

WORKDIR /dist
COPY --from=CPP_BUILDER /angel_libtorch.zip ./
COPY --from=JAVA_BUILDER /app/target/*.jar ./

VOLUME /output

CMD [ "/bin/sh", "-c", "cp ./* /output" ]
