#ifndef ML_FRESH_UPLOAD_CORE_H
#define ML_FRESH_UPLOAD_CORE_H
#include <stdint.h>
#include <string.h>
/* Private cross-object contract. Never added to VitaGL's vbo descriptor. */
typedef struct {
 uint32_t binding,storage,size,allocator,last_frame,mapped,access,usage,error;
} MLFreshStore;
typedef struct {MLFreshStore store;uint32_t epoch,owner,valid;} MLFreshLease;
typedef struct {
 MLFreshLease lease,grant;
 uint32_t pending,epoch,owner,size,usage;
} MLFreshState;
enum {ML_FRESH_OTHER,ML_FRESH_BIND,ML_FRESH_ERROR,ML_FRESH_DATA};
static inline int ml_fresh_request(uint32_t target,uint32_t size,uint32_t usage){
 return target==0x8892u && size>0 && size<=32768u && usage==0x88e0u;
}
static inline int ml_fresh_ready(const MLFreshStore *s,uint32_t size,uint32_t usage){
 return s->binding && s->storage && s->size==size && s->allocator &&
  s->last_frame==0xffffffffu && !s->mapped && s->access==0x88b9u && s->usage==usage && !s->error;
}
static inline void ml_fresh_before(MLFreshState *s,uint32_t op,uint32_t tid,uint32_t epoch,
 uint32_t target,uint32_t size,uint32_t data,uint32_t usage,const MLFreshStore *before){
 s->grant.valid=0;s->pending=0;
 if(s->lease.valid && (s->lease.owner!=tid || s->lease.epoch!=epoch))s->lease.valid=0;
 if(op==ML_FRESH_BIND || op==ML_FRESH_ERROR)return;
 MLFreshLease previous=s->lease;s->lease.valid=0;
 if(op!=ML_FRESH_DATA || !ml_fresh_request(target,size,usage) || !before)return;
 if(!data && before->binding && !before->storage && !before->mapped && !before->error){
  s->pending=before->binding;s->size=size;s->usage=usage;s->epoch=epoch;s->owner=tid;
 }else if(data && previous.valid && ml_fresh_ready(before,size,usage) &&
           !memcmp(&previous.store,before,sizeof(*before))){
  s->grant=previous;s->grant.valid=1;
 }
}
static inline void ml_fresh_after(MLFreshState *s,uint32_t tid,uint32_t epoch,const MLFreshStore *after){
 s->grant.valid=0;
 if(s->pending){
  if(tid==s->owner && epoch==s->epoch && after && after->binding==s->pending &&
      ml_fresh_ready(after,s->size,s->usage)){
   s->lease.store=*after;s->lease.owner=tid;s->lease.epoch=epoch;s->lease.valid=1;
  }else s->lease.valid=0;
 }
 s->pending=0;
}
/* Consume once; this is called by the actual driver before replacing a store. */
static inline int ml_fresh_take(MLFreshState *s,uint32_t tid,uint32_t epoch,
 const MLFreshStore *actual,uint32_t size,uint32_t data,uint32_t usage){
 MLFreshLease grant=s->grant;s->grant.valid=0;
 if(!grant.valid || grant.owner!=tid || grant.epoch!=epoch || !actual ||
    !ml_fresh_request(0x8892u,size,usage) || !ml_fresh_ready(actual,size,usage) ||
    memcmp(&grant.store,actual,sizeof(*actual)))return 0;
 /* Current candidate is restricted to native source pointers outside low GPU
  * storage. Actual pool classification is additionally checked in the driver. */
 if(data<0x81000000u || data>=0xa0000000u || size>0xa0000000u-data)return 0;
 if(actual->storage>=0x81000000u || size>0x81000000u-actual->storage)return 0;
 return 1;
}
#endif
