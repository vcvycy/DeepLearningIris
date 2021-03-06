# Triplet Loss 实现
# (1) batch_hard_triplet_loss  只返回最难的三元组loss 值的和
# (2) batch_all_triplet_loss   返回所有非easy(||a-p|| - ||a-n||+margin>0) 的三元组的loss
"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf

# (*) 返回embeddings 两两之间的距离
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

# (*) 返回embeddings 经过eweight加权后，两两之间的距离
def distances_with_weight(embeddings,eweight):
    # 参数说明：
    # (1) embeddings 为特征向量(二维展开为一维)
    # (2) eweight 为特征向量的权值
    eweight0 = tf.expand_dims(eweight,0)
    eweight1 = tf.expand_dims(eweight,1)
    eweight_3d = eweight0 * eweight1           # eweight_3d[i,j]表示第i个emb和第j个emb的权值
    embed0 = tf.expand_dims(embeddings,0)
    embed1 = tf.expand_dims(embeddings,1)
    embed_weighted = embed1 * eweight_3d       # embed_weighted[i,j] 表示第第i个emb和第j个emb加完权值后的特征值
    embed_l2 = tf.nn.l2_normalize(embed_weighted, 2)  # 加权后的向量L2归一化。大小batch*bacth*n
    embed_l2_transpose = tf.transpose(embed_l2, [1, 0, 2])
    # 求加权后的embedding距离
    dist = tf.reduce_sum(embed_l2 * embed_l2 + embed_l2_transpose * embed_l2_transpose - 2 * embed_l2 * embed_l2_transpose, 2)
    dist =  tf.maximum(dist, 0.0)
    return dist

# 返回boolean 所有二元组，如果二者label一致则为True
def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask

# 返回batch_size* batch_size 的boolean mask，表示某个二元组，是否是label不同
def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

# 返回n*n*n的mask，表示某个三元组是否合法. 输入 batch_size ，输出 batch*size^3
def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

# 返回所有合法的triplet 三元组,以及其Loss值
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # (1)返回两两之间的欧式距离, batch_size* batch_size ,
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1) 在第2维后，添加一个1的维度
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    # 巧妙的方法:  triplet_loss[i,j,k] = d[i,j,1] - d[i,1,k] +margin = d[i,j] - d[i,k] +margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)             # mask(n*n*n) ，当且仅当 label(a)=label(p)!=label(n) 且a!=p时，mask = True mask[a,p,n]=1
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # 即 当 dist(a,n) - dist(a,p) > margin时，此三元组Loss为0 ，不参与训练
    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)          # positive_triplets ：即loss 值>0 的合法triplet
    num_valid_triplets = tf.reduce_sum(mask)                       # 所有valid的 triplets (即 label(a)=label(p)!=label(n) 且a!=p时)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)  # 合法(loss>0) 的三元组占的比例

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)       # 合法的三元组 的平均loss值

    return triplet_loss, fraction_positive_triplets

def batch_all_triplet_loss_semi(labels, embeddings,eweight,margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # (1)返回两两之间的欧式距离, batch_size* batch_size ,
    # Get the pairwise distance matrix
    pairwise_dist = distances_with_weight(embeddings,eweight) #_pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1) 在第2维后，添加一个1的维度
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    # 巧妙的方法:  triplet_loss[i,j,k] = d[i,j,1] - d[i,1,k] +margin = d[i,j] - d[i,k] +margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)             # mask(n*n*n) ，当且仅当 label(a)=label(p)!=label(n) 且a!=p时，mask = True mask[a,p,n]=1
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # 即 当 dist(a,n) - dist(a,p) > margin时，此三元组Loss为0 ，不参与训练
    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)          # positive_triplets ：即loss 值>0 的合法triplet
    num_valid_triplets = tf.reduce_sum(mask)                       # 所有valid的 triplets (即 label(a)=label(p)!=label(n) 且a!=p时)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)  # 合法(loss>0) 的三元组占的比例

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)       # 合法的三元组 的平均loss值

    return triplet_loss, fraction_positive_triplets

# 最难的三元组：
# (1) 任意一个embedding 作为anchor
# (2) 找出距离其最近的negtive 样例
# (3) 找出距离其最远的postive 样例
# (4) 构成一组三元组
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # 两两之间的距离
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)


    # 获取Anchor - Postive 的Mask，表示二元组是否合法
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # 不合法的anchor-postive二元组，距离清空为0
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # 每个anchor，找出距离其最大的 postive 样例
    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # 每个anchor 找出距离其最小的 negtive 样例
    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.expand_dims(tf.reduce_max(pairwise_dist, axis=1), 1)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # 最近的negtive 样例：shape (batch_size,)
    hardest_negative_dist = tf.expand_dims(tf.reduce_min(anchor_negative_dist, axis=1,),1)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # 最终的triplet loss 值
    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_semi(labels, embeddings, eweight,margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # 两两之间的距离
    # Get the pairwise distance matrix
    pairwise_dist = distances_with_weight(embeddings,eweight)


    # 获取Anchor - Postive 的Mask，表示二元组是否合法
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # 不合法的anchor-postive二元组，距离清空为0
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # 每个anchor，找出距离其最大的 postive 样例
    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # 每个anchor 找出距离其最小的 negtive 样例
    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.expand_dims(tf.reduce_max(pairwise_dist, axis=1), 1)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # 最近的negtive 样例：shape (batch_size,)
    hardest_negative_dist = tf.expand_dims(tf.reduce_min(anchor_negative_dist, axis=1,),1)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # 最终的triplet loss 值
    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss