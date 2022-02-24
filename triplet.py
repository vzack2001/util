import tensorflow as tf

def _masked_maximum(data, mask, dim=1):
    """ Computes the axis wise maximum over chosen elements.
        Args:
            data: 2-D float `Tensor` of shape `[n, m]`.
            mask: 2-D Boolean `Tensor` of shape `[n, m]`.
            dim: The dimension over which to compute the maximum.

        Returns:
            masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
    """
    mask = tf.cast(mask, tf.float32)
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums

def _masked_minimum(data, mask, dim=1):
    """ Computes the axis wise minimum over chosen elements.
        Args:
            data: 2-D float `Tensor` of shape `[n, m]`.
            mask: 2-D Boolean `Tensor` of shape `[n, m]`.
            dim: The dimension over which to compute the minimum.

        Returns:
            masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
    """
    mask = tf.cast(mask, tf.float32)
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums

@tf.function
def pairwise_distance(feature, squared=False, L2_normalize=False, dtype=tf.float32):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    if L2_normalize:
        feature = tf.math.l2_normalize(feature, axis=1)

    pairwise_distances_squared = (
        tf.math.add(
            tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
            tf.math.reduce_sum(
                tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
            ),
        )
        - 2.0 * tf.matmul(feature, tf.transpose(feature))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(pairwise_distances_squared)

    return tf.cast(pairwise_distances, dtype=dtype)

@tf.function
def angular_distance(feature, dtype=tf.float32):
    """ Computes the angular distance matrix.
            output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])
        Args:
            feature: 2-D Tensor of size `[number of data, feature dimension]`.
        Returns:
            angular_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # normalize input
    feature = tf.math.l2_normalize(feature, axis=1)

    # create adjaceny matrix of cosine similarity
    angular_distances = 1 - tf.matmul(feature, feature, transpose_b=True)

    # ensure all distances > 1e-16
    angular_distances = tf.maximum(angular_distances, 0.0)

    return tf.cast(angular_distances, dtype=dtype)

@tf.function
def triplet_semihard_loss( y_true, y_pred, margin=1.0, distance_metric="L2", ):
    r"""Computes the triplet loss with semi-hard negative mining.
        Usage:
            >>> y_true = tf.convert_to_tensor([0, 0])
            >>> y_pred = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
            >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric="L2")
            <tf.Tensor: shape=(), dtype=float32, numpy=2.4142137>

            >>> # Calling with callable `distance_metric`
            >>> distance_metric = lambda x: tf.linalg.matmul(x, x, transpose_b=True)
            >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric=distance_metric)
            <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
        Args:
            y_true: 1-D integer `Tensor` with shape `[batch_size]` of
                multiclass integer labels.
            y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
                be l2 normalized.
            margin: Float, margin term in the loss definition.
            distance_metric: `str` or a `Callable` that determines distance metric.
                Valid strings are "L2" for l2-norm distance,
                "squared-L2" for squared l2-norm distance,
                and "angular" for cosine similarity.

                A `Callable` should take a batch of embeddings as input and
                return the pairwise distance matrix.
        Returns:
            triplet_loss: float scalar with dtype of `y_pred`.
    """
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    batch_size = tf.size(labels)

    # Reshape label tensor to [batch_size, 1].
    labels = tf.reshape(labels, [batch_size, 1])

    # Build pairwise squared distance matrix
    if distance_metric == "L2":
        pdist_matrix = pairwise_distance(
            precise_embeddings, squared=False, L2_normalize=True
        )
    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance(
            precise_embeddings, squared=True, L2_normalize=True
        )
    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)
    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))

    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])

    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])),
    )

    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(tf.cast(mask, dtype=tf.dtypes.float32), axis=1, keepdims=True),
            0.0,
        ),
        [batch_size, batch_size],
    )

    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
        #_masked_maximum(pdist_matrix, adjacency), [1, batch_size]
    )

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = adjacency - tf.linalg.diag(tf.ones([batch_size]))
    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss

@tf.function
def triplet_loss( y_true, y_pred, margin=1.0, distance_metric="L2", ):
    r""" Computes the triplet loss
    """
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    precise_embeddings = tf.cast(embeddings, tf.float32)

    # Reshape label tensor to [batch_size, 1].
    batch_size = tf.size(labels)
    labels = tf.reshape(labels, [batch_size, 1])

    # Build pairwise squared distance matrix
    if distance_metric == "L2":
        pdist_matrix = pairwise_distance(
            precise_embeddings, squared=False, L2_normalize=True
        )
    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance(
            precise_embeddings, squared=True, L2_normalize=True
        )
    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)
    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))

    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    # negatives_outside: smallest D_an.
    min_outside = _masked_minimum(pdist_matrix, adjacency_not)

    # negatives_inside: largest D_ap.
    max_inside = _masked_maximum(pdist_matrix, adjacency)

    loss_mat = tf.math.add(margin, max_inside - min_outside)

    loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(loss_mat, 0.0)
        ),
        tf.cast(batch_size, tf.float32),
    )

    return loss


if __name__ == "__main__":
    import numpy as np
    from mylib import print_ndarray

    size = (128, 128)
    labels = np.random.randint(0, high=2, size=size[0])
    feature = np.random.normal(loc=1, scale=1, size=size)
    print_ndarray(f'\nlabels (y_true)', labels)
    print_ndarray(f'feature (y_pred)', feature)

    angular_dist = angular_distance(feature)
    print_ndarray(f'angular_dist', angular_dist)

    pairwise_dist = pairwise_distance(feature, squared=False)
    print_ndarray(f'pairwise_dist', pairwise_dist)

    pairwise_dist = pairwise_distance(feature, squared=False, L2_normalize=True)
    print_ndarray(f'pairwise_dist', pairwise_dist)

    t = triplet_loss(labels, feature, margin=1.0, distance_metric='squared-L2', )  # 'L2', 'squared-L2', 'angular'
    print(f'\ntriplet_loss: {t:.3f}')

    t = triplet_semihard_loss(labels, feature, margin=1.0, distance_metric='squared-L2', )  # 'L2', 'squared-L2', 'angular'
    print(f'\ntriplet_semihard_loss: {t:.3f}')

    pass